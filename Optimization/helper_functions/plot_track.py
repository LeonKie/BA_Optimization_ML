



plt.rcParams["mathtext.fontset"] = 'stix' # math fonts
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid


def draw_courve(curve=None,showPlot=True):
    if (curve is None):
        createImg()
        coef=getcoef([listx,listy],20)
        print("\nCoefficient X:\n",coef[0].T,"\n--------\nCoefficient Y:\n",coef[1])
        c=set_Curve(list(coef[0]),list(coef[1]))
        if showPlot:
            plotcurve(c)
    else:
        if showPlot:
            c=curve
            plotcurve(c)
    return c

def plotcurve(curve):
    global steps;
    time=np.linspace(0,1,steps);
    plt.clf
    pxy=curve.createList()
    
    fig, axs = plt.subplots(1,2)
    axs[0].plot(curve.fx(time[0]),curve.fy(time[0]),"g*",markersize=10)
    axs[0].plot(curve.fx(time[-1]),curve.fy(time[-1]),"ro",markersize=10)
    axs[0].plot(curve.fx(time),curve.fy(time),marker="o",markersize=3)
    axs[0].set_title("Kurve without equal Spacing")
    axs[0].legend(["Start","End","Curve"])

    X=([x[0] for x in pxy])
    Y=([y[1] for y in pxy])

    axs[1].plot(X[0],Y[0],"g*",markersize=10)
    axs[1].plot(X[-1],Y[-1],"ro",markersize=10)
    axs[1].plot(X,Y,marker="o",markersize=3)
    axs[1].set_title("Kurve with equal Spacing")
    axs[1].legend(["Start","End","Curve"])
    plt.show()


def drawTrack(raceTrack):
    