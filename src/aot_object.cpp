// LLVM IR text in, an object file out: the one thing the AOT compiler cannot do
// in Dream without writing an instruction selector and an optimizer of its own.
//
// It is here rather than in a program of its own so that building a native
// executable needs nothing beyond the toolchain that was already installed: the
// VM carries LLVM for its JIT, and no clang or llc has to be found. It shares no
// state with the JIT, and the AOT's output owes nothing to the VM -- this is a
// service the compiler calls, not a part of what it compiles.
//
// See dreams/docs/aot.md.

#include "jit.hpp"

#include <memory>
#include <mutex>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#ifdef DREAM_AOT_CROSS
extern "C" {
void LLVMInitializeX86TargetInfo();
void LLVMInitializeX86Target();
void LLVMInitializeX86TargetMC();
void LLVMInitializeX86AsmPrinter();
void LLVMInitializeX86AsmParser();
void LLVMInitializeAArch64TargetInfo();
void LLVMInitializeAArch64Target();
void LLVMInitializeAArch64TargetMC();
void LLVMInitializeAArch64AsmPrinter();
void LLVMInitializeAArch64AsmParser();
}
#endif

namespace dream {

bool aot_object(std::string_view ir, int opt_level, std::string* object, std::string* error) {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
#ifdef DREAM_AOT_CROSS
        // The other machines `dreams --aot --target` writes for (`DREAM_AOT_CROSS` in CMake).
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmPrinter();
        LLVMInitializeX86AsmParser();
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
        LLVMInitializeAArch64AsmParser();
#endif
    });

    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(ir.data(), ir.size()), "aot",
                                                   /*RequiresNullTerminator=*/false);
    std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
    if (!module) {
        *error = "line " + std::to_string(diagnostic.getLineNo()) + ": " + diagnostic.getMessage().str();
        return false;
    }
    std::string problems;
    llvm::raw_string_ostream problem_stream(problems);
    if (llvm::verifyModule(*module, &problem_stream)) {
        *error = problems;
        return false;
    }

    // The module's own triple wins, so that the compiler is what says what it
    // generated for; only the host's target is linked in today, and asking for
    // another is an answer from the registry rather than a crash.
    llvm::Triple triple(module->getTargetTriple().str().empty()
                            ? llvm::Triple(llvm::sys::getDefaultTargetTriple())
                            : module->getTargetTriple());
    std::string lookup;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget("", triple, lookup);
    if (!target) {
        *error = lookup;
        return false;
    }
    const auto level = opt_level <= 0   ? llvm::CodeGenOptLevel::None
                       : opt_level == 1 ? llvm::CodeGenOptLevel::Less
                                        : llvm::CodeGenOptLevel::Default;
    // Static and not position-independent: the executable writer lays out a
    // fixed address space and resolves the relocations itself, and PIC would only
    // add a GOT for it to build.
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        triple, "generic", "", llvm::TargetOptions(), llvm::Reloc::Static, llvm::CodeModel::Small, level));
    if (!machine) {
        *error = "no target machine for " + triple.str();
        return false;
    }
    module->setTargetTriple(triple);
    module->setDataLayout(machine->createDataLayout());

    if (opt_level > 0) {
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
        llvm::PassBuilder builder(machine.get());
        builder.registerModuleAnalyses(mam);
        builder.registerCGSCCAnalyses(cgam);
        builder.registerFunctionAnalyses(fam);
        builder.registerLoopAnalyses(lam);
        builder.crossRegisterProxies(lam, fam, cgam, mam);
        const auto o = opt_level == 1 ? llvm::OptimizationLevel::O1 : llvm::OptimizationLevel::O2;
        builder.buildPerModuleDefaultPipeline(o).run(*module, mam);
    }

    llvm::SmallVector<char, 0> bytes;
    llvm::raw_svector_ostream stream(bytes);
    llvm::legacy::PassManager emit;
    if (machine->addPassesToEmitFile(emit, stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        *error = "this target cannot write an object file";
        return false;
    }
    emit.run(*module);
    object->assign(bytes.data(), bytes.size());
    return true;
}

}  // namespace dream
