import gzip
import csv
import linear
import numpy as np
import lpputils
import time
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

#converts time into time in seconds
def toSec(timedate):
    date, time = timedate.split()
    time = time.split(".")[0]
    time2 = sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":")))
    return time2

#returns 0 if it's a weekday
#and 1 if it is weekend
def weekday(year, month, day):
    wday = datetime.date(year, month, day).weekday()
    if wday <= 4:
         wday = 0
    else:
        wday = 1
    return wday

#returns 1 if date is a holiday, else 0
def praznik(month, day):
    wday = 0
    praznikiJan = [1, 2]
    praznikiFeb = [8]
    praznikiApr = [9, 27]
    praznikiMaj = [1, 2]
    praznikiJun = [25]
    praznikiAvg = [15]
    praznikOkt = [31]
    praznikiNov = [1]
    praznikiDec = [25, 26]

    if month == 1 and day == any(praznikiJan) or \
        month == 2 and day == any(praznikiFeb) or \
        month == 4 and day == any(praznikiApr) or \
        month == 5 and day == any(praznikiMaj) or \
        month == 6 and day == any(praznikiJun) or \
        month == 8 and day == any(praznikiAvg) or \
        month == 10 and day == any(praznikOkt) or \
        month == 11 and day == any(praznikiNov) or \
        month == 12 and day == any(praznikiDec):
            wday = 1
    return wday

#returns 1 if date is a school holiday, else 0
def pocitnice(month, day):
    pocitnice = 0
    pocitniceFeb = [20, 21, 22, 23, 24]
    pocitniceJunij = [28, 29, 30, 31]
    pocitniceOkt = [29, 30]
    pocitniceNov = [2]
    pocitniceDec = [24, 27, 28, 31]
    if month == 2 and day == any(pocitniceFeb) or \
        month == 6 and day == any(pocitniceJunij) or \
        month == 7 or \
        month == 8 or \
        month == 10 and day == any(pocitniceOkt) or \
        month == 11 and day == any(pocitniceNov) or \
        month == 12 and day == any(pocitniceDec):
         pocitnice = 1
    return pocitnice

if __name__ == "__main__":
    #read train data file
    f = gzip.open("train.csv.gz", "rt", encoding="utf8")
    reader = csv.reader(f, delimiter="\t")
    next(reader) #skip legend

    #reads csv file with weather data
    vreme = {}
    f = open("padavine_lj_2012.csv", "rt", encoding="utf8")
    readerVreme = csv.reader(f, delimiter="\t")
    next(readerVreme) #skip legend
    for line in readerVreme:
        row = line[0].strip().split(";")
        #print(row)
        day, month, year = row[0].split(" ")
        date = year + "-" + month + "-" + day
        sneg = row[-8]
        if sneg == "  ":
            sneg = 0
        sneg = int(sneg)
        if sneg > 0:
            sneg = 1
        dez = row[-9]
        if dez == "    ":
            dez = 0
        if isinstance(dez, str):
            dez = int(dez.split(",")[0])
        #if dez > 0:
            #dez = 1
        #weather is saved in a dictionary, where key is date
        vreme[date] = [sneg, dez]


    #dictionary with data for each bus linea
    dict = {}
    #create y
    ydict = {}
    for d in reader:
        app = []
        #time of departure in sec
        timeSec = toSec(d[-3])
        app.append(timeSec)

        #hour/2 of departure
        timeHour = int(timeSec/3600)
        app.append(timeHour/2)

        #gets date in three (year, month, day) variables instead of one
        timedate = d[-3]
        date, timeHMS = timedate.split()
        year, month, day = (int(x) for x in date.split('-'))

        #adds weather to x
        if date in vreme.keys():
            sneg, dez = vreme[date]
            app.append(sneg)
            #app.append(dez)
        else:
            app.append(0)
            #app.append(0)

        #day in a week
        day = weekday(year, month, day)
        app.append(day)

        #school holidays (1-true, 0-false)
        pocit = pocitnice(month, day)
        app.append(pocit)

        #holidays (1-true, 0-false)
        prazniki = praznik(month, day)
        app.append(prazniki)

        #traffic by hours
        hour = timeSec/3600
        rushHour1 = 0
        rushHour2 = 0
        nightTime = 0
        if hour >= 22 or hour < 6:
            nightTime = 1
        elif hour >= 6 and hour < 9:
            rushHour1 = 1
        elif hour >=14.5 and hour <16.5:
            rushHour2 = 1

        app.append(rushHour1)
        app.append(rushHour2)
        app.append(nightTime)

        #bus line
        #last station isn't last station anymore but Route
        lineNmb = int(d[2])
        lastStation = d[3]
        maybe = lineNmb, lastStation

        #appends all the previous data to dictionary
        if maybe in dict.keys():
            dict[maybe].append(app)
        else:
            dict[maybe] = [app]

        #calculates and appends travel time to y for testing
        travelTime = int(lpputils.tsdiff(d[-1], d[-3]))
        if maybe in ydict.keys():
            ydict[maybe].append(travelTime)
        else:
            ydict[maybe] = [travelTime]


    #make the arrays correct dimensions
    for key in dict:
        dict[key] = np.array(dict[key])
        ydict[key] = np.array(ydict[key])

    #calculate the average error of all the models
    error = 0
    for key in dict:
        X = dict[key]
        y = ydict[key]
        #split in 11 folds to get months (more or less)
        kf = KFold(n_splits=11, shuffle=False) # Define the split - into 11 folds
        kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

        #caluclate the error of one model
        tmpError = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #do linear regression for each groups
            lr = linear.LinearLearner(lambda_=1.)
            napovednik = lr(X_train, y_train)

            y_napovedani = np.array([])
            for enu, test in enumerate(X_test):
                y_napovedani = np.append(y_napovedani, napovednik(test))

            tmpError = tmpError + mean_absolute_error(y_napovedani, y_test)

        tmpError = tmpError/11
        error = error + tmpError
    #average the error of all the models
    print(error/len(dict.keys()))

    #priprava rezultatov za oddajo na učilnico
    #save all linear models
    models = {}
    for key in dict:
        X = dict[key]
        y = ydict[key]
        lr = linear.LinearLearner(lambda_=1.)
        napovednik = lr(X, y)
        #print(napovednik)
        models[key] = napovednik


    #read train data file
    f = gzip.open("test.csv.gz", "rt", encoding = "utf8")
    reader = csv.reader(f, delimiter="\t")
    next(reader) #skip legend
    data = [d for d in reader]

    fo = open("rezultati2.txt", "wt")
    for d in data:
        #do all the sam things as previously, except on train data
        app = []
        #time of departure in sec
        timeSec = toSec(d[-3])
        app.append(timeSec)

        timeHour = int(timeSec/3600)
        app.append(timeHour/2)

        timedate = d[-3]
        date, timeHMS = timedate.split()
        year, month, day = (int(x) for x in date.split('-'))

        if date in vreme.keys():
            sneg, dez = vreme[date]
            app.append(sneg)
            #app.append(dez)
        else:
            app.append(0)
            #app.append(0)

        #day in a week
        day = weekday(year, month, day)
        app.append(day)

        pocit = pocitnice(month, day)
        app.append(pocit)

        prazniki = praznik(month, day)
        app.append(prazniki)

        #traffic by hours
        hour = timeSec/3600
        rushHour1 = 0
        rushHour2 = 0
        nightTime = 0
        if hour >= 22 or hour < 6:
            nightTime = 1
        elif hour >= 6 and hour < 9:
            rushHour1 = 1
        elif hour >=14.5 and hour <16.5:
            rushHour2 = 1

        app.append(rushHour1)
        app.append(rushHour2)
        app.append(nightTime)

        #bus line
        lineNmb = int(d[2])
        lastStation = d[3]
        maybe = lineNmb, lastStation

        #write predicitions to file
        fo.write(lpputils.tsadd(d[-3], models[maybe](app)) + "\n")

    fo.close()
