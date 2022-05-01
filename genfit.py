#!/bin/python
# Geschrieben 10/2020, Henry Korhonen henryk@ethz.ch, basierend auf Matlabskripten von Martin Willeke. Hinweise, Bemerkungen und Vorschläge bitte an henryk@ethz.ch.

# importieren von libraries bzw. Funktionen.
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from math import sqrt
import matplotlib.pyplot as plt

xy = pd.read_table('test_lin.txt', names=['x','y'], sep=r'\s+') # Lesen der Daten, erstellen eines Dataframes. Als Separator kommt hier eine unbestimmte Anzahl Leerschläge in Frage. Andernfalls "sep" anpassen.

y = xy['y'] # Relevante Daten aus dem Dataframe extrahieren. Achtung: "names" in pd.read_table gibt der ersten Spalte den Namen x und der zweiten y. Unbedingt sicherstellen, dass die richtigen Daten extrahiert werden!
x = xy['x']

N = len(y) # Anzahl Datenpunkte ermitteln. Diese Anzahl wird weiter unten benötigt, um die Anzahl Freiheitsgrade zu bestimmen.

#slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)




def func(omega, taud, etazero, q, n): # Funktion definieren, die gefittet werden soll. Alle Variablen müssen im Argument (das in Klammern) angegeben werden.

    return etazero/( 1+(taud*omega)**q)**((1-n)/q) # Eta(Omega)=Etazero/( 1+(taud*Omega)^q)^((1-n)/q))

popt, pcov = curve_fit(func, x, y, maxfev=50000) # fitten der Daten
# popt, pcov = curve_fit(func, x, y, bounds=([alower, blower], [aupper, bupper])) # fitten der Daten mit Eingrenzung der Regressionskoeffizienten
pstd = np.sqrt(np.diag(pcov)) # Standardabweichung der Regressionskoeffizienten. Nota bene: auf der Diagonalen von pcov stehen die Varianzen der Regressionskoeffizienten.

# Vertrauensintervall berechnen
alpha = 0.05 # m%-Vertrauensintervall: m = 100*(1-alpha)
p = len(popt)
dof = max(0,N-p) # Anzahl Freiheitsgrade (nota bene: das hängt von der Anzahl Regressionskoeffizienten in der Fitfunktion ab (siehe def func(...) oben)
tinv = stats.t.ppf(1.0-alpha/2., dof) # Student-T-Faktor ermitteln

print('Anzahl Freiheitsgrade: {0}\nAnzahl Messungen: {1}\n=================================='.format(dof, N))
for i, regkoeff,var in zip(range(N), popt, np.diag(pcov)): # Hier werden alle Regressionskoeffizienten mit den entsprechenden Vertrauensintervallen ausgegeben.
    sigma = var**0.5
    print('Parameter {0}: {1} \n Vertrauensintervall: [{2}  {3}] \n Standardabweichung: {4} \n =================================='.format(i+1, regkoeff, regkoeff - sigma*tinv, regkoeff + sigma*tinv, sigma))

plt.plot(x,y,'b.',label='data') # Daten plotten
plt.plot(x, func(x, *popt), 'r-', label='fit: taud=%5.3f, etazero=%5.3f, q=%5.3f, n=%5.3f' % tuple(popt)) # Gefittete Funktion plotten. Auch hier muss angepasst werden, wenn mehr Parameter genutzt werden sollen (siehe def func(...) oben)


plt.xlabel('x') # Labels setzen
plt.ylabel('y')
plt.legend() # Legende generieren



plt.savefig("dateiname.pdf") # Plot als PDF-Datei speichern.
plt.savefig("dateiname.png") # Plot als PNG-Datei speichern.

plt.show() # Plot anzeigen
