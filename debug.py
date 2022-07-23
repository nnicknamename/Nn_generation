from sys import stdout

class Debug():
    def __init__(self):
        self.lastLine=''

    def writeUpdate(self,text):
        stdout.write("\r" + " " * len(self.lastLine) + "\r" + text)
        self.lastLine=text
    
    def write(self,text):
        stdout.write(text)
        self.lastLine=''
        