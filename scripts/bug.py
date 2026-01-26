from rich.traceback import install
# show locals means local variables are shown in the traceback
install(show_locals=False)

def do_something():
    x = 10
    y = 0
    x/y

do_something()
