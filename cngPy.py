import numpy
import math

# In this implementation, assume that continous data are drawn from
# gaussian distribution

class DataProcessing(object):
    def __init__(self, trainPath, testPath):
        self.trainingPath = trainPath
        # Dictionary for storing the parsed training and testing data
        # the structure is as follow
        # { (attribute one, attribute two....): label,
        #   (attribute one, attribute two....): label...}
        self.trainingDataDic = dict()
        self.testingDataDic = dict()
        self.testingPath = testPath
        # This dictionary is used to store all the possible values of
        # discrete attribute. for example, gender has male and female
        # the attribute is represented by index in the training data
        self.discreteAttrDic = {1:{"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                                             "State-gov", "Without-pay", "Never-worked"},
                                3:{"Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                                             "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th",
                                             "10th", "Doctorate", "5th-6th", "Preschool"},
                                5:{"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                                    "Married-spouse-absent", "Married-AF-spouse"},
                                6:{"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                                    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                                    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                                    "Armed-Forces"},
                                7:{"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"},
                                8:{"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"},
                                9:{"Female", "Male"},
                                13:{"United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                                    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                                    "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
                                    "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
                                    "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
                                    "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
                                    "Holand-Netherlands"}}
        self.labelSet = {"<=50K", ">50K"}
        # Index of column of all discrete attributes and continuely attributes
        # respectively
        self.discreteAttrColumn = [1, 3, 5, 6, 7, 8, 9, 13]
        self.continAttrColumn = [0, 2, 4, 10, 11, 12]
        # Attribute that needs to be normalized
        self.normalizeAttr = [10]
        # Dictionary for storing the means, variances and conditional P of
        # discrete and continuous attributes of training data
        # the structure of dictionary is as follow:
        # { tuple(index of attribute, label): probability or tuple(mean, variance) }
        self.trainDiscreteAttr = dict()
        self.trainConAttr = dict()
        # Dictionary for storing the P of label calculated from training data
        self.trainLabelP = dict()
        # A every small number to avoid gaussian distribution
        # divide by zero when variance equal to 0
        self.epilson = 10**-9
        self.accuracy = 1.0
        # Set log0 to a very small number as a negative infinite
        self.negativeInfinite = -(10**9)

    def setDiscreteAttrDict(self, dict):
        self.discreteAttrDic = dict

    def setLabel(self, labelSet):
        self.labelSet = labelSet

    def setDiscreteAttrColumn(self, l):
        self.discreteAttrColumn = l

    def setContinAttrColumn(self, l):
        self.continAttrColumn = l

    def setEilson(self, newEpilson):
        self.epilson = newEpilson

    def readFile(self, filePath, dataDict):
        with open(filePath, "r") as f:
            data = f.readlines()
            for line in data:
                if line != "\n" and line != "" and line != " ":
                    trainData = self.parseData(line)
                    dataDict[tuple(trainData[:-1])] = trainData[-1]

    def parseData(self, line):
        return line[:-1].split(", ")

    def readTestFile(self):
        self.readFile(self.testingPath, self.testingDataDic)

    def readTrainFile(self):
        self.readFile(self.trainingPath, self.trainingDataDic)

    # Calculate the mean and variances for continuous attributes given the label
    def calDataMeanAndVariance(self, dataDict, attr, label):
        # Calculate the mean
        sum, labelNum = 0, 0
        for data in dataDict:
            if dataDict[data] == label:
                sum = float(data[attr]) + sum
                labelNum = labelNum + 1
        mean = sum/labelNum

        # Calculate the variance
        var = 0
        for data in dataDict:
            if dataDict[data] == label:
                var = (float(data[attr]) - mean)**2 + var
        var = var/(labelNum - 1)
        return (mean, var)

    # Calculate the P(X = x|Y = y) for continue case.
    # x is the value of a continuous attribute
    def gaussian(self, x, mean, variance):
        j = 2*math.pi*(variance + self.epilson)
        k = j**0.5
        l = 1/k
        a = (x - mean) ** 2
        b = 2 * (variance + self.epilson)
        c = -a/b
        d = math.exp(c)
        return l*d

    # Calculate the probability of discrete attributes given label
    # Just count the number of times this attribute occur given label
    def getDiscreteAttrP(self, dataDict, attr, subAttr, label):
        occur = 0
        labelNum = 0
        for data in dataDict:
            # Ignore unknown values
            if data[attr] != "?" and dataDict[data] == label:
                labelNum = labelNum + 1
                if data[attr] == subAttr:
                    occur = occur + 1
        return occur/labelNum

    # Normalize a given attributes value
    def normalize(self, attr, dataDict):
        value = []
        for data in dataDict:
            value.append(float(data[attr]))
        value = sorted(value)
        low, high = value[0], value[-1]

        newDictionary = dict()
        for data in dataDict:
            newData = list(data)
            newData[attr] = (float(newData[attr])/(high - low))*1
            newDictionary[tuple(newData)] = dataDict[data]
            #print(newDictionary[tuple(newData)])
        return newDictionary


    # Check whether attribute is continuous or discrete
    def isDiscreteAttr(self, attr):
        return attr.isdigit()

    def calPofLabel(self, dataDict):
        for label in self.labelSet:
            self.trainLabelP[label] = 0

        for data in dataDict:
            self.trainLabelP[dataDict[data]] = self.trainLabelP[dataDict[data]] + 1

        for label in self.trainLabelP:
            #print("%s, %f" %(label, self.trainLabelP[label]/len(dataDict)))
            self.trainLabelP[label] = self.trainLabelP[label]/len(dataDict)

    # Main function for training data
    def train(self):
        # Read the process training file
        print("Start reading files...")
        self.readTrainFile()

        # Get the mean and variance of different continuous attributes
        print("Calculating the mean and variance for continuous attributes...")
        for label in self.labelSet:
            for attr in self.continAttrColumn:
                mean, variance = self.calDataMeanAndVariance(self.trainingDataDic, attr, label)
                # Assign mean and variance of a given attribute and label to the dictionary
                # which will be used to calculate the conditional P of testing data
                print("label: %s, attr: %i, mean: %f, var: %f" %(label, attr, mean, variance))
                self.trainConAttr[(attr, label)] = (mean, variance)

        print("Calculating the conditional probability for discrete attributes...")
        # Get the conditional P of different discrete attributes
        for label in self.labelSet:
            for attr in self.discreteAttrColumn:
                for subAttr in self.discreteAttrDic[attr]:
                    self.trainDiscreteAttr[(subAttr, label)] = self.getDiscreteAttrP(self.trainingDataDic,
                                                                                     attr, subAttr, label)
        # Calculate the P of training label
        print("Calculating the probability of labels...")
        self.calPofLabel(self.trainingDataDic)

        print("finish!")

    # Use the training data to predict a given testing data
    def predict(self, attributes):
        # Get the log(P) of label
        logResult = dict()
        for label in self.labelSet:
            logResult[label] = math.log(self.trainLabelP[label])
            #print(logResult[label])

            for i in range(len(attributes)):
                if i in self.continAttrColumn:
                    mean, variance = self.trainConAttr[(i, label)]
                    #print("curren x: %f" %float(attributes[i]))
                    p = self.gaussian(float(attributes[i]), mean, variance)
                    if p == 0:
                        logResult[label] = logResult[label] + self.negativeInfinite
                    else:
                        logResult[label] = logResult[label] + math.log(p)
                else:
                    if attributes[i] != "?":
                        if self.trainDiscreteAttr[(attributes[i], label)] == 0:
                            logResult[label] = logResult[label] + self.negativeInfinite
                        else:
                            logResult[label] = logResult[label] + \
                                              math.log(self.trainDiscreteAttr[(attributes[i], label)])
                        #print("disc pro: %f" %(logResult[label]))
                    else:
                        continue

        # Return the label that maximize the naive bayes classifier
        temp = dict()
        for label in logResult:
            temp[logResult[label]] = label

        #print(sorted(temp.keys()), temp)
        return temp[sorted(temp.keys())[-1]]


    def selectTrain(self):
        # Read in data set
        self.readTrainFile()

        # Divide the data sets into several small pieces



    # Main function for testing data
    def test(self):
        # Read and parse the testing file
        print("Reading test file...")
        self.readTestFile()

        # Calculate the P for given data
        print("predicting...")
        error = 0
        for data in self.testingDataDic:
            predict = self.predict(data)
            print("predict: %s, true: %s" % (predict, self.testingDataDic[data]))
            if predict != self.testingDataDic[data][:-1]:
                error = error + 1

        accuracy = (len(self.testingDataDic) - error)/len(self.testingDataDic)
        print("============================================")
        print("Accuracy: %f" %accuracy)
        print("Error: %f" %(error/len(self.testingDataDic)))


'''test = DataProcessing("H:\\701machinelearning\\Program\\Gaussian Naive Bayes\\dataSet\\adult.data.txt",
                      "H:\\701machinelearning\\Program\\Gaussian Naive Bayes\\dataSet\\adult.test.txt")
test.train()
test.test()'''