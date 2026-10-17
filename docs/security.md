# Security boundaries

Dream programs run with the host process's permissions. The VM is not an
operating-system sandbox: `std.ffi` calls native code, and the IO, OS and
network modules deliberately expose host resources. Green-process heap
isolation does not restrict these capabilities. Compiler evaluation and macros
also execute code; compiling an untrusted project is not a safe inspection
operation. Run hostile projects in an externally restricted process or VM,
without secrets and with filesystem, network, CPU and memory limits.

Image validation is a memory-safety boundary. The interpreter and JIT rely on
validated operands and frame sizes. Validation must visit every executable
operand, including indexing fallbacks, and reject argument counts larger than
the declared frame. The node graph must be acyclic; its cycle check enumerates
each edge once rather than rebuilding wide child lists for each edge. These
checks do not impose a total execution-time or host-resource budget.

HTTP/1 readers require CRLF and reject invalid header names and control bytes
in field values. Body framing accepts one decimal Content-Length or one
Transfer-Encoding containing exactly `chunked`. Duplicate framing fields,
combined Transfer-Encoding and Content-Length, unsupported encodings, signed
or nondecimal lengths, malformed chunk terminators, and incomplete trailers
are errors. Duplicate identical lengths are deliberately rejected too.
Trailers are limited to 64 KiB including line endings; leading blank lines
consume the header budget. This is stricter than the previous reader, so peers
sending bare LF or ambiguous framing must be corrected.

These changes are a targeted hardening pass, not certification that the entire
VM and standard library are free from vulnerabilities. Native extensions are
trusted code. HTTP/2, application authorization, outbound request policy, and
complete denial-of-service resistance require their own threat models and
review. In particular, the HTTP encoding APIs accept application-built request
and response values: applications must not treat arbitrary untrusted values
as validated wire metadata.

Regression coverage lives in `dream/tests/test_main.cpp` (malformed frames,
indexing operands and wide graphs), `dream/tests/programs/http_hardening.dr`
(HTTP framing and parser limits), and `dream/tests/fuzz_image.sh` (mutated
images). Run the VM unit suite and end-to-end suite with the current compiler;
the latter checks both interpreter and JIT behavior.
