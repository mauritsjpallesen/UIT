connect() # defined in boot.py

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 80))
s.listen(5)

def respond(conn, status_code):
    if status_code == 200:
        conn.send('HTTP/1.1 200 OK\n')
        body = "Ok"
    else:
        conn.send('HTTP/1.1 400 Not Found\n')
        body = "Not Found"

    conn.send('Content-Type: text/html\n')
    conn.send('Connection: close\n\n')
    conn.send(body)

while True:
    conn, addr = s.accept()
    print('Connection from %s' % str(addr))
    path = str(conn.readline()).split(' ')[1]

    if path == '/red':
        red_led.value(1) # defined in boot.py
        green_led.value(0) # defined in boot.py
        respond(conn, 200)
    elif path == '/green':
        red_led.value(0)
        green_led.value(1)
        respond(conn, 200)
    else:
        respond(conn, 404)

