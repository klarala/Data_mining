import numpy as np
import math
from scipy.optimize import fmin_l_bfgs_b


def load(name):
    """
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke)
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """

    verjetnostRazreda1 = 1 / (1 + np.exp(-x.dot(theta)))

    return verjetnostRazreda1


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """

    cost = 0
    for m in range(len(X)):
        cost = cost + ( y[m] * np.log(h(X[m], theta)) + (1 - y[m]) * np.log(1 - h(X[m], theta)) )

    regularizacija = 0
    for i in range(len(theta)):
        regularizacija = regularizacija + theta[i] ** 2
    regularizacija = lambda_ * regularizacija


    return -cost/len(X) + regularizacija


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    #konstanta = 0.000000001
    gradient = 0
    for m in range(len(X)):
        gradient = gradient + (y[m] - h(X[m], theta)) * X[m]

    return -gradient/len(X) + lambda_ * theta


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    numGradient = []
    h = 0.0000001
    for i in range(len(theta)):
        theta2 = np.copy(theta)
        theta2[i] = theta2[i] + h
        numGradient.append((cost(theta2, X, y, 0) - cost(theta, X, y, 0)) / h)

    return numGradient + lambda_ * theta


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri uÄenju.
    To je napaÄen naÄin ocenjevanja uspeÅ¡nosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=7):
    #vrstni red shuffla
    shuffleOrder = np.arange(len(X))
    np.random.shuffle(shuffleOrder)
    shuffledX = []
    shuffledy = []
    for i in range(len(X)):
        shuffledX.append(X[shuffleOrder[i]])
        shuffledy.append(y[shuffleOrder[i]])

    #naredimo tabelo z k deli mnozice podatkov
    petina = int(len(X)/k)
    ostanekNedeljivo = int(len(X)%k)
    #izmenicno napovedujemo rezultate vsakic na drugem delu
    napovedi = []
    for i in range(k):
        #print(i*petina, (i+1)*petina)
        testniX = []
        ucniX = []
        ucniy = []
        #zacetek in konec nastavimo tako, da deluje tudi pri nedeljivih stevilih
        zacetek = 0
        konec = petina + ostanekNedeljivo
        for j in range(k):
            #ce ni i, potem je to del mnozice, ki so ucni primeri
            if j != i:
                ucniX.extend(shuffledX[zacetek:konec])
                ucniy.extend(shuffledy[zacetek:konec])
            #ce je enako i je to testni del mnozice
            else:
                testniX.extend(shuffledX[zacetek:konec])
            zacetek = konec
            konec = konec + petina
        classifier = learner(np.array(ucniX), np.array(ucniy))
        #print(len(testniX))
        for test in testniX:
            napovedi.append(classifier(test))

    #un shuffle napovedi
    unshuffledNapovedi = np.copy(napovedi)
    for i in range(len(napovedi)):
        unshuffledNapovedi[shuffleOrder[i]] = napovedi[i]

    return np.array(unshuffledNapovedi)


def CA(real, predictions):
    # ... dopolnite (naloga 3)
    #nov array ki dodeli primer razredu z vecjo verjetnostjo
    recalculatedPredictions = []
    for verjetnost in predictions:
        razred1, razred0 = verjetnost
        if razred0 > razred1:
            recalculatedPredictions.append(0)
        else:
            recalculatedPredictions.append(1)
    #primerjava real in predictions arraya
    difference = 0
    for i in range(len(predictions)):
        difference = difference + abs(real[i] - recalculatedPredictions[i])

    return difference/len(real)


def AUC(real, predictions):
    # ... dopolnite (dodatna naloga)
    #mannwitney test
    pass


if __name__ == "__main__":
    # Primer uporabe
    lambda_ = 1.0

    X, y = load('reg.data')

    learner = LogRegLearner(lambda_)
    classifier = learner(X, y) # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    print(napoved)

    res = test_learning(LogRegLearner(lambda_), X, y)
    print ("Tocnost test_learning:", CA(y, res)) #argumenta sta pravi razredi, napovedani

    res = test_cv(learner, X, y)
    print ("Tocnost test_cv:", CA(y, res)) #argumenta sta pravi razredi, napovedani
