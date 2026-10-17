#!/bin/sh
# std.tls, end to end, from the root of the repository.
#
# Two halves. `tls.dr` is a server and its clients inside one VM -- the same
# program portable.py runs unchanged on Linux, macOS and Windows -- so it can
# only say that this VM agrees with itself. The other half is the reason this
# script exists: the VM against a TLS stack that is not its own, OpenSSL's
# `s_server` and `s_client`, in both directions and at both protocol versions
# it speaks. That half is skipped where there is no `openssl` command.
set -u

dream=${dream:-build-dream/bin/dream}
dreams=${dreams:-build/dreams.dream}
here=dream/tests/tls
tmp=$(mktemp -d)
pids=""
cleanup() {
    for p in $pids; do kill "$p" 2>/dev/null; done
    rm -rf "$tmp"
}
trap cleanup EXIT

fail=0
pass=0
check() {  # name expected actual
    if [ "$2" = "$3" ]; then
        pass=$((pass + 1))
    else
        echo "FAIL $1: expected [$2], got [$3]"
        fail=$((fail + 1))
    fi
}

"$dream" "$dreams" "$here/tls.dr" -L mind -o "$tmp/tls.dream" >"$tmp/build" 2>&1 || {
    echo "tls: the suite did not compile"; sed 's/^/    /' "$tmp/build"; exit 1; }
timeout 120 "$dream" "$tmp/tls.dream" "$here" || exit 1

# std.http on top: HTTPS at both versions, ALPN choosing between them.
"$dream" "$dreams" "$here/http.dr" -L mind -o "$tmp/http.dream" >"$tmp/build" 2>&1 || {
    echo "tls: the HTTP suite did not compile"; sed 's/^/    /' "$tmp/build"; exit 1; }
timeout 120 "$dream" "$tmp/http.dream" "$here" || exit 1

if ! command -v openssl >/dev/null 2>&1; then
    echo "tls interop: skipped -- no openssl command"
    exit 0
fi

# A program that is one side of a conversation with OpenSSL.
cat >"$tmp/peer.dr" <<DR
import std.console;
import std.error;
import std.file;
import std.io;
import std.net;
import std.num;
import std.os;
import std.streams;
import std.tls;

let main! = match os.args! () {
    ["client", port, dir] => {
        let sock = net.connect! "127.0.0.1" (match num.parse_int port { () => 0, n => n });
        tls.connect! sock %{ :host => "localhost", :ca => file.read! (dir + "/ca.pem") }
        streams.write_all! sock "hello from dream\n"
        console.print! (io.read! sock 4096)
        io.close! sock
    },
    ["probe", port, dir] => {
        let outcome = try! {
            let sock = net.connect! "127.0.0.1" (match num.parse_int port { () => 0, n => n });
            tls.connect! sock %{ :host => "localhost", :ca => file.read! (dir + "/ca.pem") }
            io.close! sock
            :ok
        } catch e { error.kind e };
        console.print! (to_string outcome)
    },
    ["server", dir] => {
        let listener = net.listen! 0;
        console.print! (to_string (net.port! listener))
        let conn = net.accept! listener;
        tls.accept! conn %{ :identity => file.read! (dir + "/server.p12"), :password => "dream" }
        streams.write_all! conn ("dream heard: " + io.read! conn 100)
        io.close! conn
    },
    _ => (),
};
DR
"$dream" "$dreams" "$tmp/peer.dr" -L mind -o "$tmp/peer.dream" >"$tmp/build" 2>&1 || {
    echo "tls interop: the peer did not compile"; sed 's/^/    /' "$tmp/build"; exit 1; }

free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

for version in -tls1_2 -tls1_3; do
    # The VM as the client: s_server sends each line back, reversed.
    port=$(free_port)
    openssl s_server -accept "$port" "$version" -cert "$here/server.pem" -key "$here/server.key" \
        -quiet -naccept 1 -rev >"$tmp/s_server.log" 2>&1 &
    pids="$pids $!"
    sleep 0.5
    got=$(timeout 20 "$dream" "$tmp/peer.dream" client "$port" "$here" 2>&1 | head -1)
    # `-rev` sends each line back reversed.
    check "dream client, openssl s_server $version" "maerd morf olleh" "$got"

    # The VM as the server, and s_client verifying it against the test CA.
    timeout 20 "$dream" "$tmp/peer.dream" server "$here" >"$tmp/port" 2>&1 &
    pids="$pids $!"
    sleep 0.5
    port=$(head -1 "$tmp/port")
    got=$(echo "from s_client" | timeout 20 openssl s_client "$version" -connect "127.0.0.1:$port" \
            -CAfile "$here/ca.pem" -servername localhost -verify_return_error -quiet 2>/dev/null)
    check "openssl s_client $version, dream server" "dream heard: from s_client" "$got"
done

# Stapled OCSP: s_server hands over a fixed response with its certificate, and
# the VM must refuse the one that says "revoked" and accept the one that says
# "good". The VM never fetches a response itself, so a staple is the only way
# one reaches it.
for case in "revoked :certificate_revoked" "server :ok"; do
    set -- $case
    port=$(free_port)
    openssl s_server -accept "$port" -cert "$here/$1.pem" -key "$here/$1.key" -status_file "$here/$1.ocsp" \
        -quiet -naccept 1 -www >"$tmp/s_server.log" 2>&1 &
    pids="$pids $!"
    sleep 0.5
    got=$(timeout 20 "$dream" "$tmp/peer.dream" probe "$port" "$here" 2>&1 | head -1)
    check "a stapled OCSP response for the $1 certificate" "$2" "$got"
done

# HTTP/2 against a client that is not this VM's: curl, built with nghttp2,
# negotiating h2 by ALPN. Skipped where curl cannot speak HTTP/2.
if command -v curl >/dev/null 2>&1 && curl --version | grep -q HTTP2; then
    timeout 30 "$dream" "$tmp/http.dream" serve "$here" >"$tmp/http_port" 2>&1 &
    pids="$pids $!"
    sleep 0.5
    port=$(head -1 "$tmp/http_port")
    got=$(timeout 10 curl -s --http2 --cacert "$here/ca.pem" -w ' %{http_version}' "https://localhost:$port/")
    check "curl over HTTP/2, dream server" "HTTP/2 2" "$got"
    got=$(head -c 300000 /dev/zero | tr '\0' 'x' | timeout 10 curl -s --http2 --cacert "$here/ca.pem" --data-binary @- "https://localhost:$port/echo" | wc -c | tr -d ' ')
    check "curl uploading 300 KB over HTTP/2" "300000" "$got"
else
    echo "http/2 interop: skipped -- no curl with HTTP/2"
fi

echo "tls interop: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
