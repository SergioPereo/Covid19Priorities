import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import math

df = pd.read_csv('covid-colonias-mensual.csv')
df = df.dropna()

df_priority_colonies = pd.read_csv('colonias-de-atencion-prioritaria-covid-kioscos.csv')

df_population = df_priority_colonies[['clave_colonia','poblacion_total']]

df_actives = df_priority_colonies[['clave_colonia','activos']]
max_actives = df_priority_colonies[['activos']].sum()

df_with_population = df.merge(df_population,left_on='clave_colonia',right_on='clave_colonia')

df_with_population = df_with_population.dropna()

max_actives = max_actives[0]



gb = df_with_population.groupby('clave_colonia')
colonies = [gb.get_group(x).sort_values(by='fecha_referencia') for x in gb.groups]

slopes = np.array([])
colony_names = np.array([])
colony_id = np.array([])

def make_plots(x_labels, y, regression, title, file_path, x_label, y_label):
    plt.figure()
    plt.scatter(x_labels, y)
    plt.plot(x_labels, regression)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.savefig(file_path)
    plt.close()   

for colony in colonies:
    if len(colony.index) > 2:
        x_labels = colony['fecha_referencia'].tolist()
        x = list(range(len(x_labels)))
        positives = colony['casos_positivos'].tolist()
        sum_positives = [sum(positives[:i+1]) for i,value in enumerate(positives)]
        y = [x/colony['poblacion_total'].iloc[0] for x in sum_positives]
        #print(sum_positives, colony['poblacion_total'].iloc[0])
        #print(x)
        #print(positives)
        #print(sum_positives)
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        regression = list(map(lambda x:slope*x+intercept,x))
        slopes = np.append(slopes,[slope])
        colony_names = np.append(colony_names,[colony['nombre_colonia'].iloc[0]])
        colony_id = np.append(colony_id,[colony['clave_colonia'].iloc[0]])
        #make_plots(x_labels,y,regression,colony['nombre_colonia'].iloc[0],f"colonies_pgrowth_and_approx/{colony['clave_colonia'].iloc[0]}.jpg","Fecha","Casos positivos")
max_slopes = slopes.sum()
process_data = pd.DataFrame({'Growth': slopes, 'nombre_colonia': colony_names, 'clave_colonia': colony_id})

actives_mapping = interp1d([0, max_slopes],[0, max_actives])

process_data['optimized_actives'] = process_data.apply(lambda row: math.floor(actives_mapping(row['Growth'])), axis=1)

comparing_data = process_data.merge(df_actives,left_on='clave_colonia',right_on='clave_colonia')

comparing_data['actives_comparison'] = comparing_data.apply(lambda row: row['activos']-row['optimized_actives'], axis=1)


print(comparing_data)
print(process_data.sort_values(by='Growth', ascending=False))
print(max_slopes, max_actives)