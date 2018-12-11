class fileHelper(object):
    def __init__(self, fileName):
        self.file = open(fileName, "a+")

    # def outputResult(self, results):
    #     for x in results:
    #         self.file.write(str(x)+"\n")

    def outputResult(self, results):
        for i in range(results.shape[0]):
            self.file.write(str(results[i,0])+"\n")

    def fileClose(self):
        self.file.close()
