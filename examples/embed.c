/* Embedding the Dawn VM from C: load an image, register a host module, run it.
 *
 *   cc examples/embed.c -Iinclude -Lbuild/lib -ldawn -o embed
 *   dreams program.dr --host-module host -o program.dream
 *   LD_LIBRARY_PATH=build/lib ./embed program.dream
 *
 * `--host-module host` is what tells the compiler that `import host;` resolves
 * to a module this program registers at run time rather than to Dream source.
 * `dreams` is an image; run it as `<vm> <dreams.dream> --host-module host ...`.
 */
#include <stdio.h>
#include <string.h>

#include "dream/dream.h"

/* A host function Dawn code reaches as `host.shout!`. Arguments arrive already
 * forced, because the registration marks them strict. */
static dream_result host_shout(dream_process* p, const dream_value* args, uint32_t argc,
                              dream_value* out) {
    uint32_t len = 0;
    const char* s = dream_value_string(args[0], &len);
    if (!s) {
        *out = dream_make_error(p, "type_error", "shout! wants a string");
        return DREAM_BAD;
    }
    for (uint32_t i = 0; i < len; ++i) {
        int c = (unsigned char)s[i];
        putchar(c >= 'a' && c <= 'z' ? c - 32 : c);
    }
    putchar('\n');
    *out = dream_make_unit();
    return DREAM_OK;
}

/* A variadic host function, reached as `host.total!`. Registering it with an
 * arity of DREAM_VARIADIC means it receives exactly the arguments its call site
 * passed -- `host.total! 1 2 3` arrives as argc == 3 -- with every one forced.
 *
 * The flip side of variadic in a curried language is that such a function can
 * never be partially applied: `f a b` and a half-finished `f a b c` differ only
 * in the application node, so there is nothing to wait for. */
static dream_result host_total(dream_process* p, const dream_value* args, uint32_t argc,
                               dream_value* out) {
    dream_integer sum = 0;
    for (uint32_t i = 0; i < argc; ++i) {
        if (dream_value_type(args[i]) != DREAM_TYPE_INTEGER) {
            *out = dream_make_error(p, "type_error", "total! wants integers");
            return DREAM_BAD;
        }
        sum += dream_value_integer(args[i]);
    }
    *out = dream_make_integer(p, sum);
    return DREAM_OK;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: embed <image.dream>\n");
        return 2;
    }

    dream_vm* vm = dream_vm_new();

    const char* names[] = {"shout!", "total!"};
    const uint32_t arities[] = {1, DREAM_VARIADIC};
    const uint32_t strict[] = {1, 0}; /* ignored for a variadic member */
    const dream_native_fn fns[] = {host_shout, host_total};
    dream_vm_register_module(vm, "host", names, arities, strict, fns, 2);

    char err[256];
    if (!DREAM_IS_OK(dream_vm_load_file(vm, argv[1], err, sizeof err))) {
        fprintf(stderr, "load failed: %s\n", err);
        dream_vm_free(vm);
        return 1;
    }

    dream_result r = dream_vm_run(vm, NULL);
    printf("result: %s\n", dream_vm_result_text(vm));
    printf("reductions: %llu\n", (unsigned long long)dream_vm_reductions(vm));

    int failed = dream_vm_failed(vm);
    dream_vm_free(vm);
    return DREAM_IS_OK(r) && !failed ? 0 : 1;
}
