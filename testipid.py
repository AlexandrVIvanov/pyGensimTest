import socketserver


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            r = self.request.recv(8192)
            if b"\r\n\r\n" in r or b"\n\n" in r:
                break
            if not r:
                return
        print("-----\r\n" + r.decode() + "-----")
        self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n")
        self.request.sendall(b"OK")
        return


if __name__ == "__main__":
    HOST, PORT = "0.0.0.0", 8038
    socketserver.ForkingTCPServer.allow_reuse_address = True
    server = socketserver.ForkingTCPServer((HOST, PORT), MyTCPHandler)
    server.allow_reuse_address = True
    server.serve_forever()
