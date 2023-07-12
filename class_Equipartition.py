from dbm.dumb import error
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import copy
import time, timeit
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
from matplotlib import cm
from numpy.ma.core import sqrt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from IPython.display import display, clear_output

class Equipartition:

    def __init__(self,polygon,number_of_regions):
        self.poligon = polygon
        self.number_of_regions = number_of_regions
    

    def punto_mas_cercano(puntos_iniciales, punto, pesos_iniciales):
        c=0
        dist=(puntos_iniciales[0][0]-punto[0])**2+(puntos_iniciales[0][1]-punto[1])**2+pesos_iniciales[0]
        for i in range(0,np.shape(puntos_iniciales)[0],1):
            if (puntos_iniciales[i][0]-punto[0])**2+(puntos_iniciales[i][1]-punto[1])**2+pesos_iniciales[i]< dist:
                c=i
                dist = (puntos_iniciales[i][0]-punto[0])**2+(puntos_iniciales[i][1]-punto[1])**2+pesos_iniciales[i]
        return(c) 

    def puntos_mas_cercanos(puntos_iniciales, punto, pesos_iniciales):
        c=0
        v=[]
        dist=(puntos_iniciales[0][0]-punto[0])**2+(puntos_iniciales[0][1]-punto[1])**2+pesos_iniciales[0]
        for i in range(0,np.shape(puntos_iniciales)[0],1):
            if (puntos_iniciales[i][0]-punto[0])**2+(puntos_iniciales[i][1]-punto[1])**2+pesos_iniciales[i]< dist:
                c=i
                dist = (puntos_iniciales[i][0]-punto[0])**2+(puntos_iniciales[i][1]-punto[1])**2+pesos_iniciales[i]
        v.append(c)
        for i in range(c+1,np.shape(puntos_iniciales)[0],1):
            if (puntos_iniciales[i][0]-punto[0])**2+(puntos_iniciales[i][1]-punto[1])**2+pesos_iniciales[i] == dist:
                v.append(i)        
        return(v) 

    def puntos_en_region_n(puntos_iniciales, points,j, pesos_iniciales):
        v=[]
        for i in range(0,np.shape(points)[0]):
            if j in Equipartition.puntos_mas_cercanos(puntos_iniciales, points[i], pesos_iniciales):
                v.append(i)  
        return(v) 

    def segment_segment_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
        d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
        if d:
            uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
            uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
        else:
            return
        if not(0 <= uA <= 1 and 0 <= uB <= 1 ): 
            return
        x = Ax1 + uA * (Ax2 - Ax1)
        y = Ay1 + uA * (Ay2 - Ay1)
    
        return x, y

    def segment_line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
        d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
        if d:
            uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
            uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
        else:
            return
        if not(0 <= uA <= 1 and 0 <= uB  ):
            return
        x = Ax1 + uA * (Ax2 - Ax1)
        y = Ay1 + uA * (Ay2 - Ay1)
    
        return x, y   

    def weighted_ridge(pt1,pt2,l1,l2):
        a=(-l1+l2+(sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2))**2)/(2*(sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)))
        uni=sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)
        ridge=np.array([pt1[0]+a*((pt2[0]-pt1[0])/uni),pt1[1]+a*((pt2[1]-pt1[1])/uni)])
        dir = pt1-pt2
        dir_perp=np.array([dir[1],-dir[0]])
        vertice=ridge+dir_perp
        return np.array([ridge,vertice])                                                                               

    def vect_intersect(line1,line2):
        d = (line2[1][1] - line2[0][1]) * (line1[1][0] - line1[0][0]) - (line2[1][0] - line2[0][0]) * (line1[1][1] - line1[0][1])
        if d:
            uA = ((line2[1][0] - line2[0][0]) * (line1[0][1] - line2[0][1]) - (line2[1][1] - line2[0][1]) * (line1[0][0] - line2[0][0])) / d
            uB = ((line1[1][0] - line1[0][0]) * (line1[0][1] - line2[0][1]) - (line1[1][1] - line1[0][1]) * (line1[0][0] - line2[0][0])) / d
            x = line1[0][0] + uA * (line1[1][0] - line1[0][0])
            y = line1[0][1] + uA * (line1[1][1] - line1[0][1])
            return np.array([x, y]) 
        else:
            return 

    def triple_intersect(pt1,pt2,pt3,l1,l2,l3): 
        return Equipartition.vect_intersect(Equipartition.weighted_ridge(pt1,pt2,l1,l2),Equipartition.weighted_ridge(pt2,pt3,l2,l3))


    def graph_regiones(regiones,sitios):
        plt.figure(figsize=(5, 5)) 
        paleta = cm.get_cmap('nipy_spectral', 8)
        for n in range(0,len(regiones),1):
            plt.scatter(regiones[n][:,0], regiones[n][:,1], c='k')
            for i in range(-1,np.shape(regiones[n])[0]-1,1):
                plt.plot([regiones[n][i][0],regiones[n][i+1][0]],[regiones[n][i][1],regiones[n][i+1][1]],c='blue')
            plt.annotate("{}".format(n), (sitios[n][0],sitios[n][1]), textcoords="offset points", xytext=(0,5)) 
        plt.scatter(sitios[:,0], sitios[:,1], c='r')    
        plt.show() 

    def graph_only_regiones(regiones):
        plt.figure(figsize=(5, 5)) 
        paleta = cm.get_cmap('nipy_spectral', 8)
        for n in range(0,len(regiones),1):
            plt.scatter(regiones[n][:,0], regiones[n][:,1], c='k')
            for i in range(-1,np.shape(regiones[n])[0]-1,1):
                plt.plot([regiones[n][i][0],regiones[n][i+1][0]],[regiones[n][i][1],regiones[n][i+1][1]],c='blue')  
        plt.show()     

    def graph_regiones_filled(regiones,sitios):
        plt.figure(figsize=(5, 5)) 
        paleta = cm.get_cmap('nipy_spectral', 8)
        for n in range(0,len(regiones),1):
            plt.scatter(regiones[n][:,0], regiones[n][:,1], c='k')
            for i in range(-1,np.shape(regiones[n])[0]-1,1):
                plt.plot([regiones[n][i][0],regiones[n][i+1][0]],[regiones[n][i][1],regiones[n][i+1][1]],c='blue')
            plt.fill(regiones[n][:,0],regiones[n][:,1], color=paleta((n+1)/len(regiones))) 
        plt.show()    

    def area(R):
        respuesta=0
        region = np.append(R, np.array([R[0]]),  axis=0)
        for i in range(np.shape(R)[0]):
            respuesta = respuesta + (region[i][0]*region[i+1][1]-region[i+1][0]*region[i][1])/2
        return respuesta

    def centroide(R):
        centrox=0
        centroy=0
        region = np.append(R, np.array([R[0]]),  axis=0)
        for i in range(np.shape(R)[0]):
            centrox = centrox + (region[i][0]+region[i+1][0])*(region[i][0]*region[i+1][1]-region[i+1][0]*region[i][1])
            centroy = centroy + (region[i][1]+region[i+1][1])*(region[i][0]*region[i+1][1]-region[i+1][0]*region[i][1])
        areatotal=Equipartition.area(R)
        centrox = centrox/(6*areatotal)
        centroy = centroy/(6*areatotal)
        return np.array([[centrox,centroy]])


    def graph_regiones_centroides(regiones,sitios):
        plt.figure(figsize=(5, 5)) 
        plt.title("Centroidal Voronoi Diagram")   
        paleta = cm.get_cmap('nipy_spectral', 8)
        for n in range(0,len(regiones),1):
            plt.scatter(regiones[n][:,0], regiones[n][:,1], c='k')
            for i in range(-1,np.shape(regiones[n])[0]-1,1):
                plt.plot([regiones[n][i][0],regiones[n][i+1][0]],[regiones[n][i][1],regiones[n][i+1][1]],c='blue')
            plt.annotate("{}".format(n), (sitios[n][0],sitios[n][1]), textcoords="offset points", xytext=(0,5)) 
            centro = Equipartition.centroide(regiones[n])
            plt.scatter(centro[:,0], centro[:,1], c='g', marker='^')
        plt.scatter(sitios[:,0], sitios[:,1], c='r')    
        plt.show()     

    def graph_only_regiones_centroides(regiones):
        plt.figure(figsize=(5, 5))
        plt.title("Final Polygon Partition")   
        paleta = cm.get_cmap('nipy_spectral', 8)
        for n in range(0,len(regiones),1):
            plt.scatter(regiones[n][:,0], regiones[n][:,1], c='k')
            for i in range(-1,np.shape(regiones[n])[0]-1,1):
                plt.plot([regiones[n][i][0],regiones[n][i+1][0]],[regiones[n][i][1],regiones[n][i+1][1]],c='blue')
            centro = Equipartition.centroide(regiones[n])
            plt.annotate("{}".format(n), (centro[0][0],centro[0][1]), textcoords="offset points", xytext=(0,5))
            plt.scatter(centro[:,0], centro[:,1], c='g', marker='^')
        plt.show() 


    def weight_dist(punto,sitio,peso):
        value = -2*(punto[0]*sitio[0]+punto[1]*sitio[1]) +sitio[0]**2 + sitio[1]**2 + peso
        return value

    def weighted_voronoi(sitios,pesos):  #retorna: vertices en array(n,2), regiones (lista de listas con el número de fila de sus resp vertices), ridge_vertices (indices de los vertices que definen frontera), ridge_points (indices de sitios que definen el ridge)
        lista=[]
        length_sitios=len(sitios)
        for i in range (0,length_sitios-2,1):
            for j in range (i+1,length_sitios-1,1):
                for k in range (j+1,length_sitios,1): #para cada tripla de indices, hallamos el punto de intersección de las rectas
                    interseccion= Equipartition.triple_intersect(sitios[i],sitios[j],sitios[k],pesos[i],pesos[j],pesos[k])
                    if length_sitios==3:
                        #if poligon_path.contains_point(interseccion) == True:
                        lista.append([i,j,k,interseccion])
                    else: 
                        new_sitios=np.delete(sitios,[i,j,k],0)
                        new_pesos=np.delete(pesos,[i,j,k],0)  
                        w=True 
                        value = Equipartition.weight_dist(interseccion, sitios[i], pesos[i])
                    for q in range(len(new_sitios)):
                        w=w*(value < Equipartition.weight_dist(interseccion, new_sitios[q],new_pesos[q]))
                    if w==True:
                        lista.append([i,j,k,interseccion])
        df = pd.DataFrame(lista, columns=['l','m','n','inter']) #convertimos en dataframe indices e intersecciones
        vertices = np.vstack(df['inter'])
        regiones = [ [] for i in range(len(sitios))]
        ridge_vertices=[]
        ridge_points=np.empty([0,2], dtype=np.int32)
        for i in range(0,len(sitios),1):
            regiones[i] = df[(df['l']==i)|(df['m']==i)|(df['n']==i)].index.to_list() #vertices de cada región, numerados por su fila en la matriz de vértices
            for j in range(0,len(sitios),1):
                data=df[((df['l']==i)&(df['m']==j))|((df['l']==i)&(df['n']==j))|((df['m']==i)&(df['n']==j))]
                if len(data)==1:
                    ridge_vertices.append([-1,data.index[0]])  #ridge abierto en una dirección
                    ridge_points = np.vstack((ridge_points, np.array([[i,j]]))) #sitios que definen el ridge
                elif len(data)==2:
                    ridge_vertices.append([data.index[0],data.index[1]])   #ridge cerrado de dos interesecciones
                    ridge_points = np.vstack((ridge_points, np.array([[i,j]]))) #sitios que definen el ridge
        return vertices, regiones, ridge_vertices,ridge_points #en 0 devuelve vertices, en 1 devuelve points  

    def particion_wv(poligono, sitios, pesos):
        poligon = ConvexHull(poligono)
        poligon_path = Path( poligon.points[poligon.vertices] )
        number_of_regions = np.shape(sitios)[0]
        #pesos=np.array([0,10,0,0,0,0])
        vertices_wv, regiones_wv, ridge_vertices_wv, ridge_points_wv = Equipartition.weighted_voronoi(sitios,pesos)  #(vor.vertices)
        regiones=[]
        for i in range(0,number_of_regions,1):
            list= regiones_wv[i].copy()  #La indexación de las regiones la dan los sitios (puntos originales). Hay que corregir porque en Voronoi salen otros.  
            vert = vertices_wv[list].copy()
            rows_to_remove=[]  
            for j in range(0,np.shape(vert)[0],1):  # Aquí se quitan los puntos de cada región que están por fuera de la región acotada
                if poligon_path.contains_point(vert[j]) == False:
                    rows_to_remove.append(j)
            vert=np.delete(vert,rows_to_remove,0)
            a =  np.vstack((vert,poligono[Equipartition.puntos_en_region_n(sitios,poligono,i,pesos)]))
            regiones.append(a)
        
        for row, side in enumerate(ridge_vertices_wv): #se hace un loop sobre las líneas que forman el diagrama de Voronoi
            if side[0]!=-1:   #aquí se escogen las líneas del diagrama de Voronoi que son acotadas. Estás no tienen -1 en la primera coordenada
                v1x = vertices_wv[side[0]][0]  #vértice 1 del segmento
                v1y = vertices_wv[side[0]][1]
                v2x = vertices_wv[side[1]][0]  #vértice 2 del segmento
                v2y = vertices_wv[side[1]][1]
                for j in range(-1,np.shape(poligono)[0]-1,1):
                    w1x = poligono[j][0]   #vértice 1 del lado de la región externa
                    w1y = poligono[j][1]
                    w2x = poligono[j+1][0]   #vértice 2 del lado de la región externa
                    w2y = poligono[j+1][1]
                    if Equipartition.segment_segment_intersect(v1x,v1y,v2x,v2y,w1x,w1y,w2x,w2y) != None:
                        x , y = Equipartition.segment_segment_intersect(v1x,v1y,v2x,v2y,w1x,w1y,w2x,w2y)  #punto de intersección
                        new_intersection = np.array([[x,y]])
                        region_1_to_add = ridge_points_wv[row][0]  #el punto se le añade a las regiones con frontera el segmento
                        region_2_to_add = ridge_points_wv[row][1]
                        regiones[region_1_to_add] = np.vstack((regiones[region_1_to_add],new_intersection))
                        regiones[region_2_to_add] = np.vstack((regiones[region_2_to_add],new_intersection))
            else:
                vertex = vertices_wv[side[1]]   # Vértice del diagrama de Voronoi de donde sale una línea no acotada
                point_1= ridge_points_wv[row][0]  #puntos más cercanos al vértice cuya línea equidistante a estos dos puntos definen la línea no acotada del diagrama de Voronoi
                point_2= ridge_points_wv[row][1]
                normal = sitios[point_1]-sitios[point_2]   #vector normal a la línea no acotada. 
                dir = np.array([normal[1],-normal[0]])            #Perpendicular al vector que une a los dos puntos
                sitios_sin = np.delete(sitios,[point_1, point_2],axis=0)  #Se le quitan los dos puntos para poder ubicar el tecero más cercano (en los vértices hay tres puntos equidistantes)
                pesos_sin = np.delete(pesos,[point_1, point_2],axis=0)
                cercano= Equipartition.punto_mas_cercano(sitios_sin,vertex,pesos_sin)  #se usa la función punto más cercano pero...
                if cercano>=min(point_1,point_2):
                    cercano = cercano+1                       # al quitar los dos puntos que definen la línea se mueven las filas. Hay que corregir este error. Son los dos if
                if cercano>=max(point_1,point_2):
                    cercano = cercano+1
                if Equipartition.weight_dist(vertex+dir, sitios[cercano], pesos[cercano]) < Equipartition.weight_dist(vertex+dir, sitios[point_1], pesos[point_1]):
                    dir = -dir
                v1x = vertex[0]
                v1y = vertex[1]
                v2x = (vertex+dir)[0]
                v2y = (vertex+dir)[1]
                for j in range(-1,np.shape(poligono)[0]-1,1):
                    w1x = poligono[j][0]
                    w1y = poligono[j][1]
                    w2x = poligono[j+1][0]
                    w2y = poligono[j+1][1]
                    if Equipartition.segment_line_intersect(w1x,w1y,w2x,w2y,v1x,v1y,v2x,v2y) != None:
                        x , y = Equipartition.segment_line_intersect(w1x,w1y,w2x,w2y,v1x,v1y,v2x,v2y)
                        new_intersection = np.array([[x,y]])
                        region_1_to_add = ridge_points_wv[row][0]
                        region_2_to_add = ridge_points_wv[row][1]
                        regiones[region_1_to_add] = np.vstack((regiones[region_1_to_add],new_intersection))
                        regiones[region_2_to_add] = np.vstack((regiones[region_2_to_add],new_intersection))
        areas=[]
        perimetros=[]
        for n in range(0,len(regiones),1):
            if len(regiones[n].tolist()) != 0:
                zona = ConvexHull(regiones[n])
                regiones[n] = zona.points[zona.vertices]
                areas.append(zona.volume)
                perimetros.append(zona.area)
            else:
                areas.append(0)
                perimetros.append(0)    
        lista_respuesta = []    
        lista_respuesta.append(regiones)
        lista_respuesta.append(np.asarray(areas))
        lista_respuesta.append(np.asarray(perimetros))
        # Lista de respuesta. Devuelve en [0] las regiones, en [1] las areas, en [2] los perimetros
        return lista_respuesta #


    def internal_vertices_wv(poligono,sitios,pesos):  #retorna: vertices en array(n,2), regiones (lista de listas con el número de fila de sus resp vertices),
        poligon = ConvexHull(poligono)
        poligon_path = Path( poligon.points[poligon.vertices] )
        lista=[]
        length_sitios=len(sitios)
        for i in range (0,length_sitios-2,1):
            for j in range (i+1,length_sitios-1,1):
                for k in range (j+1,length_sitios,1): #para cada tripla de indices, hallamos el punto de intersección de las rectas
                    interseccion= Equipartition.triple_intersect(sitios[i],sitios[j],sitios[k],pesos[i],pesos[j],pesos[k])
                    if poligon_path.contains_point(interseccion) == True: #se revisa si el punto está en el interior
                        if length_sitios==3:
                            lista.append([i,j,k,interseccion])
                        else: 
                            new_sitios=np.delete(sitios,[i,j,k],0)
                            new_pesos=np.delete(pesos,[i,j,k],0)  
                            w=True 
                            value = Equipartition.weight_dist(interseccion, sitios[i], pesos[i])
                            for q in range(len(new_sitios)):
                                w=w*(value <Equipartition.weight_dist(interseccion, new_sitios[q],new_pesos[q]))
                            if w==True:
                                lista.append([i,j,k,interseccion])
       
        df = pd.DataFrame(lista, columns=['l','m','n','inter']) #convertimos en dataframe indices e intersecciones
        if len(lista)==0 :
            vertices = np.empty([0,2])
        else : 
            vertices = np.vstack(df['inter'])
        regiones = [ [] for i in range(len(sitios))]
        ridge_vertices=[]
        ridge_points=np.empty([0,2], dtype=np.int32)
        for i in range(0,len(sitios),1):
            regiones[i] = df[(df['l']==i)|(df['m']==i)|(df['n']==i)].index.to_list() #vertices de cada región, numerados por su fila en la matriz de vértices  
        return vertices, regiones #en 0 devuelve vertices internos, en 1 devuelve lista, con lista de puntos de cada región  

    def external_vertices_wv(poligono, sitios, pesos):
        number_of_regions = np.shape(sitios)[0]
        regiones=[]
        for i in range(0,number_of_regions,1):
            regiones.append(Equipartition.puntos_en_region_n(sitios,poligono,i,pesos))
        return poligono, regiones

    def intermediate_vertices_wv(poligono, sitios, pesos):
        poligon = ConvexHull(poligono)
        poligon_path = Path( poligon.points[poligon.vertices] )
        number_of_regions = np.shape(sitios)[0]
        vertices_wv, regiones_wv, ridge_vertices_wv, ridge_points_wv = Equipartition.weighted_voronoi(sitios,pesos)  #(vor.vertices)
        regiones=[]
        for i in range(0,number_of_regions,1):
            list= regiones_wv[i].copy()  #La indexación de las regiones la dan los sitios (puntos originales). Hay que corregir porque en Voronoi salen otros.  
            vert = vertices_wv[list].copy()
            rows_to_remove=[]  
            for j in range(0,np.shape(vert)[0],1):  # Aquí se quitan los puntos de cada región que están por fuera de la región acotada
                if poligon_path.contains_point(vert[j]) == False:
                    rows_to_remove.append(j)
            vert=np.delete(vert,rows_to_remove,0)
            a =  np.vstack((vert,poligono[Equipartition.puntos_en_region_n(sitios,poligono,i,pesos)]))
            regiones.append(a)

        intermediate_vertices=np.empty((0,2))
        regiones_intermediate=[ [] for i in range(len(sitios))]
        sides_intermediate=[]
        count=0
        for row, side in enumerate(ridge_vertices_wv): #se hace un loop sobre las líneas que forman el diagrama de Voronoi
            if side[0]!=-1:   #aquí se escogen las líneas del diagrama de Voronoi que son acotadas. Estás no tienen -1 en la primera coordenada
                v1x = vertices_wv[side[0]][0]  #vértice 1 del segmento
                v1y = vertices_wv[side[0]][1]
                v2x = vertices_wv[side[1]][0]  #vértice 2 del segmento
                v2y = vertices_wv[side[1]][1]
                for j in range(-1,np.shape(poligono)[0]-1,1):
                    w1x = poligono[j][0]   #vértice 1 del lado de la región externa
                    w1y = poligono[j][1]
                    w2x = poligono[j+1][0]   #vértice 2 del lado de la región externa
                    w2y = poligono[j+1][1]
                    if Equipartition.segment_segment_intersect(v1x,v1y,v2x,v2y,w1x,w1y,w2x,w2y) != None:
                        x , y = Equipartition.segment_segment_intersect(v1x,v1y,v2x,v2y,w1x,w1y,w2x,w2y)  #punto de intersección
                        new_intersection = np.array([[x,y]])
                        region_1_to_add = ridge_points_wv[row][0]  #el punto se le añade a las regiones con frontera el segmento
                        region_2_to_add = ridge_points_wv[row][1]
                        regiones[region_1_to_add] = np.vstack((regiones[region_1_to_add],new_intersection))
                        regiones[region_2_to_add] = np.vstack((regiones[region_2_to_add],new_intersection))
                        
                        intermediate_vertices = np.append(intermediate_vertices, new_intersection,axis=0)
                        regiones_intermediate[region_1_to_add].append(count)
                        regiones_intermediate[region_2_to_add].append(count)
                        sides_intermediate.append([j,j+1])
                        count=count+1
            else:
                vertex = vertices_wv[side[1]]   # Vértice del diagrama de Voronoi de donde sale una línea no acotada
                point_1= ridge_points_wv[row][0]  #puntos más cercanos al vértice cuya línea equidistante a estos dos puntos definen la línea no acotada del diagrama de Voronoi
                point_2= ridge_points_wv[row][1]
                normal = sitios[point_1]-sitios[point_2]   #vector normal a la línea no acotada. 
                dir = np.array([normal[1],-normal[0]])            #Perpendicular al vector que une a los dos puntos
                sitios_sin = np.delete(sitios,[point_1, point_2],axis=0)  #Se le quitan los dos puntos para poder ubicar el tecero más cercano (en los vértices hay tres puntos equidistantes)
                pesos_sin = np.delete(pesos,[point_1, point_2],axis=0)
                cercano= Equipartition.punto_mas_cercano(sitios_sin,vertex,pesos_sin)  #se usa la función punto más cercano pero...
                if cercano>=min(point_1,point_2):
                    cercano = cercano+1                       # al quitar los dos puntos que definen la línea se mueven las filas. Hay que corregir este error. Son los dos if
                if cercano>=max(point_1,point_2):
                    cercano = cercano+1
                if Equipartition.weight_dist(vertex+dir, sitios[cercano], pesos[cercano]) < Equipartition.weight_dist(vertex+dir, sitios[point_1], pesos[point_1]):
                    dir = -dir
                v1x = vertex[0]
                v1y = vertex[1]
                v2x = (vertex+dir)[0]
                v2y = (vertex+dir)[1]
                for j in range(-1,np.shape(poligono)[0]-1,1):
                    w1x = poligono[j][0]
                    w1y = poligono[j][1]
                    w2x = poligono[j+1][0]
                    w2y = poligono[j+1][1]
                    if Equipartition.segment_line_intersect(w1x,w1y,w2x,w2y,v1x,v1y,v2x,v2y) != None:
                        x , y = Equipartition.segment_line_intersect(w1x,w1y,w2x,w2y,v1x,v1y,v2x,v2y)
                        new_intersection = np.array([[x,y]])
                        region_1_to_add = ridge_points_wv[row][0]
                        region_2_to_add = ridge_points_wv[row][1]
                        regiones[region_1_to_add] = np.vstack((regiones[region_1_to_add],new_intersection))
                        regiones[region_2_to_add] = np.vstack((regiones[region_2_to_add],new_intersection))

                        intermediate_vertices = np.append(intermediate_vertices, new_intersection,axis=0)
                        regiones_intermediate[region_1_to_add].append(count)
                        regiones_intermediate[region_2_to_add].append(count)
                        sides_intermediate.append([j,j+1])
                        count=count+1
        
        return intermediate_vertices, regiones_intermediate, sides_intermediate

    def particion_vertices(external, intermediate, internal):
        regiones=[]
        areas=[]
        perimetros=[]
        #convex=[]
        #todos_convexos=True
        for n in range(0,len(external[1]),1):
            a=np.empty((0,2))
            a = np.append(a, external[0][external[1][n]],axis=0)
            a = np.append(a, intermediate[0][intermediate[1][n]],axis=0)
            a = np.append(a, internal[0][internal[1][n]],axis=0)
            size = np.shape(a)[0]
            if size != 0:
                zona = ConvexHull(a)
                a = zona.points[zona.vertices]
                regiones.append(a)
                areas.append(zona.volume)
                perimetros.append(zona.area)
            else:
                regiones.append([])
                areas.append(0)
                perimetros.append(0) 

        lista_respuesta = []    
        lista_respuesta.append(regiones)
        lista_respuesta.append(np.asarray(areas))
        lista_respuesta.append(np.asarray(perimetros))

        return lista_respuesta

    def optimizador(poligono,number_of_regions,areas,perimetros,peso_areas,peso_perimetros):
        poligon = ConvexHull(poligono)
        area_del_poligono = poligon.volume
        areas_mean = (area_del_poligono/number_of_regions)
        perimetros_mean = perimetros.mean()
        dist_areas = sum((areas-areas_mean)**2)
        dist_perimetros=sum((perimetros-perimetros_mean)**2)
        return dist_areas*peso_areas + dist_perimetros*peso_perimetros

    def optimizador_areas(poligono,number_of_regions,areas,perimetros,peso_areas,peso_perimetros):
        poligon = ConvexHull(poligono)
        area_del_poligono = poligon.volume
        areas_mean = (area_del_poligono/number_of_regions)
        perimetros_mean = perimetros.mean()
        dist_areas = sum(np.abs((areas-areas_mean)))
        return dist_areas*peso_areas

    def F(poligono,X):
        n=len(X)
        n=int(n/3)
        sitios = X[0:2*n].reshape(n,2)
        pesos = X[2*n:3*n].transpose()[0,:]
        part=Equipartition.particion_wv(poligono,sitios,pesos)
        a = part[1]-part[1].mean()
        b = part[2]-part[2].mean()
        return np.array([np.append(a,b)]).transpose()

    def Jac_F(poligono,X,delta):
        b = Equipartition.F(poligono,X)
        n = np.size(X)
        m = np.size(b)
        jac = np.empty((m,0))
        for column in range(0,n,1):
            base = np.zeros(n)
            base[column] = 1
            base = np.array([base]).transpose()
            cambio = (Equipartition.F(poligono, X+base*delta)-Equipartition.F(poligono, X))/delta
            jac = np.append(jac,cambio,axis=1)
        return jac

    def punto_mas_cercano_sin_pesos(puntos_iniciales, punto):
        c=0
        dist=np.linalg.norm(puntos_iniciales[0]-punto)
        for i in range(0,np.shape(puntos_iniciales)[0],1):
            if np.linalg.norm(puntos_iniciales[i]-punto)< dist:
                c=i
                dist = np.linalg.norm(puntos_iniciales[i]-punto)
        return(c)      

    def puntos_mas_cercanos_sin_pesos(puntos_iniciales, punto):
        c=0
        v=[]
        dist=np.linalg.norm(puntos_iniciales[0]-punto)
        for i in range(0,np.shape(puntos_iniciales)[0],1):
            if np.linalg.norm(puntos_iniciales[i]-punto)< dist:
                c=i
                dist = np.linalg.norm(puntos_iniciales[i]-punto)
        v.append(c)
        for i in range(c+1,np.shape(puntos_iniciales)[0],1):
            if np.linalg.norm(puntos_iniciales[i]-punto) == dist:
                v.append(i)        
        return(v) 

    def puntos_en_region_n_sin_pesos(puntos_iniciales, points,j):
            v=[]
            for i in range(0,np.shape(points)[0]):
                if j in Equipartition.puntos_mas_cercanos_sin_pesos(puntos_iniciales, points[i]):
                    v.append(i)  
            return(v)        

    def particion(poligono, sitios):
        poligon = ConvexHull(poligono)
        poligon_path = Path( poligon.points[poligon.vertices] )
        number_of_regions = np.shape(sitios)[0]
        vor = Voronoi(sitios, furthest_site=False)
        regiones=[]
        for i in range(0,number_of_regions,1):
            list= vor.regions[vor.point_region[i]].copy()  
            if -1 in list:
                list.remove(-1)   
            vert = vor.vertices[list].copy()
            rows_to_remove=[]  
            for j in range(0,np.shape(vert)[0],1): 
                if poligon_path.contains_point(vert[j]) == False:
                    rows_to_remove.append(j)
            vert = np.delete(vert,rows_to_remove,0)
            a = np.vstack((vert,poligono[Equipartition.puntos_en_region_n_sin_pesos(sitios,poligono,i)]))
            regiones.append(a) 

        for row, side in enumerate(vor.ridge_vertices): 
            if side[0]!=-1:   
                v1x = vor.vertices[side[0]][0] 
                v1y = vor.vertices[side[0]][1]
                v2x = vor.vertices[side[1]][0] 
                v2y = vor.vertices[side[1]][1]
                for j in range(-1,np.shape(poligono)[0]-1,1):
                    w1x = poligono[j][0]   
                    w1y = poligono[j][1]
                    w2x = poligono[j+1][0]   
                    w2y = poligono[j+1][1]
                    if Equipartition.segment_segment_intersect(v1x,v1y,v2x,v2y,w1x,w1y,w2x,w2y) != None:
                        x , y = Equipartition.segment_segment_intersect(v1x,v1y,v2x,v2y,w1x,w1y,w2x,w2y)  
                        new_intersection = np.array([[x,y]])
                        region_1_to_add = vor.ridge_points[row][0] 
                        region_2_to_add = vor.ridge_points[row][1]
                        regiones[region_1_to_add] = np.vstack((regiones[region_1_to_add],new_intersection))
                        regiones[region_2_to_add] = np.vstack((regiones[region_2_to_add],new_intersection))
            else:
                vertex = vor.vertices[side[1]]   
                point_1= vor.ridge_points[row][0]  
                point_2= vor.ridge_points[row][1]
                normal = vor.points[point_1]-vor.points[point_2]   
                dir = np.array([normal[1],-normal[0]])            
                sitios_sin = np.delete(sitios,[point_1, point_2],axis=0)  
                cercano= Equipartition.punto_mas_cercano_sin_pesos(sitios_sin,vertex) 
                if cercano>=min(point_1,point_2):
                    cercano = cercano+1                       
                if cercano>=max(point_1,point_2):
                    cercano = cercano+1
                if np.linalg.norm(vertex+dir-vor.points[cercano]) < np.linalg.norm(vertex+dir-vor.points[point_1]):
                    dir = -dir
                v1x = vertex[0]  
                v1y = vertex[1]
                v2x = (vertex+dir)[0]  
                v2y = (vertex+dir)[1]
                for j in range(-1,np.shape(poligono)[0]-1,1):
                    w1x = poligono[j][0]  
                    w1y = poligono[j][1]
                    w2x = poligono[j+1][0] 
                    w2y = poligono[j+1][1]
                    if Equipartition.segment_line_intersect(w1x,w1y,w2x,w2y,v1x,v1y,v2x,v2y) != None:  
                        x , y = Equipartition.segment_line_intersect(w1x,w1y,w2x,w2y,v1x,v1y,v2x,v2y)
                        new_intersection = np.array([[x,y]])
                        region_1_to_add = vor.ridge_points[row][0] 
                        region_2_to_add = vor.ridge_points[row][1]
                        regiones[region_1_to_add] = np.vstack((regiones[region_1_to_add],new_intersection))
                        regiones[region_2_to_add] = np.vstack((regiones[region_2_to_add],new_intersection))
                    
        areas=[]
        perimetros=[]
        for n in range(0,len(regiones),1):
            if len(regiones[n].tolist()) != 0:
                zona = ConvexHull(regiones[n])
                regiones[n] = zona.points[zona.vertices]
                areas.append(zona.volume)
                perimetros.append(zona.area)
            else:
                areas.append(0)
                perimetros.append(0)    
        areas = np.array(areas)
        perimetros = np.array(perimetros)    
        lista_respuesta = []    
        lista_respuesta.append(regiones)
        lista_respuesta.append(areas)
        lista_respuesta.append(perimetros)
        return lista_respuesta 


    def AP(poligono,number_of_regions,external,intermediate,internal):
        poligon = ConvexHull(poligono)
        area_del_poligono = poligon.volume
        parti=Equipartition.particion_vertices(external,intermediate,internal)
        a = parti[1]-area_del_poligono/number_of_regions
        b = parti[2]-parti[2].mean()
        return np.array([np.append(a,b)]).transpose()

    def Jac_AP(poligono,number_of_regions,external,intermediate,internal,delta):
        b = Equipartition.AP(poligono,number_of_regions,external,intermediate,internal)   
        X = np.array([internal[0].flatten()]).transpose() 
        n = np.size(X)
        m = np.size(b)
        jac = np.empty((m,0))
        for column in range(0,n,1):
            base = np.zeros(n)
            base[column] = 1
            base = np.array([base]).transpose()
            X_new = X+base*delta
            internal_new=[]
            internal_new.append(X_new.reshape((int(np.shape(X)[0]/2),2)))
            internal_new.append(internal[1])
            cambio = (Equipartition.AP(poligono,number_of_regions,external,intermediate,internal_new)-b)/delta
            jac = np.append(jac,cambio,axis=1)
        p= np.shape(intermediate[0])[0]   
        directions = np.empty((p,2))
        for row in range(0,p,1):
            inicio=external[0][intermediate[2][row][0]] 
            final=external[0][intermediate[2][row][1]]  
            dir=(final-inicio)/np.linalg.norm(final-inicio) 
            base = np.zeros(np.shape(intermediate[0])) 
            base[row] = dir
            directions[row] = dir
            intermediate_new=[]
            intermediate_new.append(intermediate[0]+base*delta)
            intermediate_new.append(intermediate[1])
            intermediate_new.append(intermediate[2])
            cambio = (Equipartition.AP(poligono,number_of_regions,external,intermediate_new,internal)-b)/delta   #derivada direccional en la dirección del vector dir
            jac = np.append(jac,cambio,axis=1)
        return [jac, directions]

    def APC(poligono,number_of_regions,external,intermediate,internal):
        poligon = ConvexHull(poligono)
        area_del_poligono = poligon.volume
        parti=Equipartition.particion_vertices(external,intermediate,internal)
        a = parti[1]-area_del_poligono/number_of_regions
        b = parti[2]-parti[2].mean()
        error_convexidad = sum(parti[1])-area_del_poligono
        vector=np.array([np.append(a,b)])
        return np.array([np.append(vector,error_convexidad)]).transpose()


    def Jac_APC(poligono,number_of_regions,external,intermediate,internal,delta):
        b = Equipartition.APC(poligono,number_of_regions,external,intermediate,internal)  
        X = np.array([internal[0].flatten()]).transpose() 
        n = np.size(X)
        m = np.size(b)
        jac = np.empty((m,0))
        for column in range(0,n,1):
            base = np.zeros(n)
            base[column] = 1
            base = np.array([base]).transpose()
            X_new = X+base*delta
            internal_new=[]
            internal_new.append(X_new.reshape((int(np.shape(X)[0]/2),2)))
            internal_new.append(internal[1])
            cambio = (Equipartition.APC(external,intermediate,internal_new)-b)/delta
            jac = np.append(jac,cambio,axis=1)
        p= np.shape(intermediate[0])[0]   #número de puntos intermedios
        directions = np.empty((p,2))
        for row in range(0,p,1):
            inicio=external[0][intermediate[2][row][0]] 
            final=external[0][intermediate[2][row][1]]  
            dir=(final-inicio)/np.linalg.norm(final-inicio) 
            base = np.zeros(np.shape(intermediate[0]))  
            base[row] = dir
            directions[row] = dir
            intermediate_new=[]
            intermediate_new.append(intermediate[0]+base*delta)
            intermediate_new.append(intermediate[1])
            intermediate_new.append(intermediate[2])
            cambio = (Equipartition.APC(poligono,number_of_regions,external,intermediate_new,internal)-b)/delta   
            jac = np.append(jac,cambio,axis=1)
        return [jac, directions]
    
    #This routine performs some adjustments on the shape of the polygon 
    def balanceo(poligono, arreglo):
        distancia_max = 0
        punto_1 = np.empty([1,2])
        punto_2 = np.empty([1,2])
        for i in range(0, len(poligono), 1):
            for j in range(i+1, len(poligono), 1):
                if (distancia_max < np.linalg.norm(poligono[i]-poligono[j])):
                    distancia_max = np.linalg.norm(poligono[i]-poligono[j])
                    punto_1 = poligono[i]
                    punto_2 = poligono[j]
        vv= (punto_1-punto_2)/np.linalg.norm(punto_1-punto_2)
        ww= np.array([-vv[1],vv[0]])
        alt_pos =0
        alt_neg =0
        for i in range(0, len(poligono),1 ):
            if (np.dot(poligono[i]-punto_1, ww) > alt_pos):
                alt_pos = np.dot(poligono[i]-punto_1, ww)
            if (np.dot(poligono[i]-punto_1, ww) < alt_neg):
                alt_neg = np.dot(poligono[i]-punto_1, ww)
        distancia_perpendicular = alt_pos - alt_neg
        const = distancia_max/distancia_perpendicular
        nuevo_arreglo=arreglo.copy()
        for j in range(0,len(arreglo),1):
            nuevo_arreglo[j]= arreglo[j] +np.dot(punto_1 - arreglo[j],ww)*(1-const)*ww
        return nuevo_arreglo    
    # This routine returns the balanced polygon into its original shape
    def desbalanceo(poligono, arreglo):
        distancia_max = 0
        punto_1 = np.empty([1,2])
        punto_2 = np.empty([1,2])
        for i in range(0, len(poligono), 1):
            for j in range(i+1, len(poligono), 1):
                if (distancia_max < np.linalg.norm(poligono[i]-poligono[j])):
                    distancia_max = np.linalg.norm(poligono[i]-poligono[j])
                    punto_1 = poligono[i]
                    punto_2 = poligono[j]
        vv= (punto_1-punto_2)/np.linalg.norm(punto_1-punto_2)
        ww= np.array([-vv[1],vv[0]])
        alt_pos =0
        alt_neg =0
        for i in range(0, len(poligono),1 ):
            if (np.dot(poligono[i]-punto_1, ww) > alt_pos):
                alt_pos = np.dot(poligono[i]-punto_1, ww)
            if (np.dot(poligono[i]-punto_1, ww) < alt_neg):
                alt_neg = np.dot(poligono[i]-punto_1, ww)
        distancia_perpendicular = alt_pos - alt_neg
        const = 1/(distancia_max/distancia_perpendicular)
        nuevo_arreglo=arreglo.copy()
        for j in range(0,len(arreglo),1):
            nuevo_arreglo[j]= arreglo[j] +np.dot(punto_1 - arreglo[j],ww)*(1-const)*ww
        return nuevo_arreglo

    def partition(polygon,number_of_regions,Regions,Areas,Perimeters):
        
        iteraciones_centroide = 100
        iteraciones_Newton = 200
        # Cuantos polígonos aleatorios y cuantas veces se repite si no alcanza solución
        numero_de_figuras=1
        numero_de_repeticiones=20
        #Se reescala el polígono para que sea igual de ancho que de alto
        balancear_poligono = True
        #balancear_poligono = False

        figura=0

        alcanzo_resultado=False

        repeticion=0

        rng_poligon = np.random.default_rng()
        random_points_poligon = rng_poligon.uniform(-10, 10,(10, 2))   # random points in 2-D
        poligon = ConvexHull(polygon)
        area_del_poligono=poligon.volume # polygon area
        poligono = poligon.points[poligon.vertices] # polygon to partitionate
        area_del_poligono=poligon.volume 

        while (alcanzo_resultado==False) and (repeticion < numero_de_repeticiones):
            
            starttime = timeit.default_timer()

            # Polygon balancing routine
            if (balancear_poligono == True) :
                poligono_original = poligono
                nuevo_poligono = Equipartition.balanceo(poligono, poligono)
                poligon = ConvexHull(nuevo_poligono)
                poligono = poligon.points[poligon.vertices] # 2-d array de puntos que definen la región externa
                area_del_poligono=poligon.volume
                
            rng_sitios = np.random.default_rng()
            random_points_poligon = rng_sitios.uniform(-10, 10,(8, 2))
            bbox = [poligon.min_bound, poligon.max_bound] #Bounding box
            poligon_path = Path( poligon.points[poligon.vertices] )
            rand_points = np.empty((number_of_regions, 2)) # Random Sites
            rand_weights= np.empty(number_of_regions) # Choose number_of_regions random weights
            for i in range(number_of_regions):
                rand_points[i] = np.array([rng_sitios.uniform(bbox[0][0], bbox[1][0]), rng_sitios.uniform(bbox[0][1], bbox[1][1])])
                rand_weights[i] = rng_sitios.uniform(bbox[0][0], bbox[1][0])  #weights are chosen arbitrarily on the same imits as box
                while poligon_path.contains_point(rand_points[i]) == False:
                    rand_points[i] = np.array([rng_sitios.uniform(bbox[0][0], bbox[1][0]), rng_sitios.uniform(bbox[0][1], bbox[1][1])])
            sitios = rand_points
            pesos = rand_weights
            slenght=len(sitios)

            coordenadas=sitios.copy()
            part = Equipartition.particion(poligono,coordenadas)


            regiones, areas, perimetros = part[0], part[1], part[2]
            error_total_normalizado= sum((part[1]/part[1].mean()-1)**2)+sum((part[2]/part[2].mean()-1)**2)
            centros = np.empty(np.shape(coordenadas))
            for i in range(len(regiones)):
                centros[i]=Equipartition.centroide(regiones[i])[0]
            coord = coordenadas    
            end_while_centros = iteraciones_centroide
            if repeticion > 0:
                end_while_centros = iteraciones_centroide - repeticion
            t=0
            while (t < end_while_centros) and (np.linalg.norm(coord-centros) >10**-4):
                part = Equipartition.particion(poligono,coordenadas)
                regiones = part[0]
                areas = part[1]
                perimetros = part[2]
                error_total_normalizado= sum((part[1]/(area_del_poligono/number_of_regions)-1)**2)+sum((part[2]/part[2].mean()-1)**2)
                centros = np.empty(np.shape(coordenadas))
                for i in range(len(regiones)):
                    centros[i]=Equipartition.centroide(regiones[i])[0]
                coord = coordenadas    
                coordenadas = centros
                t=t+1
            ssites=coordenadas.copy()

            external=Equipartition.external_vertices_wv(poligono,coordenadas,np.zeros(len(sitios))) 
            intermediate=Equipartition.intermediate_vertices_wv(poligono,coordenadas,np.zeros(len(sitios)))     
            internal=Equipartition.internal_vertices_wv(poligono,coordenadas,np.zeros(len(sitios)))
            
            # Routine to obtain cyclically ordered vertices

            regiones = Equipartition.particion_vertices(external,intermediate,internal)[0]
            todos_los_puntos = np.vstack((np.vstack((external[0],intermediate[0])),internal[0]))
            indices_regiones=[]
            
            for i in range(0, len(regiones), 1):
                bb=[]
                for l in range(0,len(regiones[i]),1):
                    bb.append(np.where((todos_los_puntos == (regiones[i][l][0], regiones[i][l][1])).all(axis=1))[0][0])
                indices_regiones.append(bb)

            # Routine to turn back the polygon to its original shape
            if (balancear_poligono == True) :
                internal_new=[]
                internal_new.append(Equipartition.desbalanceo(poligono_original, internal[0]))
                internal_new.append(internal[1])
                intermediate_new=[]
                intermediate_new.append(Equipartition.desbalanceo(poligono_original, intermediate[0]))
                intermediate_new.append(intermediate[1])
                intermediate_new.append(intermediate[2])
                external_new=[]
                external_new.append(Equipartition.desbalanceo(poligono_original, external[0]))
                external_new.append(external[1])
                
                external =external_new
                intermediate = intermediate_new
                internal = internal_new
                
                poligono = poligono_original
                poligon = ConvexHull(poligono)
                poligono = poligon.points[poligon.vertices] 
                area_del_poligono=poligon.volume

            kappa=10**-6
            
            intermediate_initial=intermediate[0]
            internal_initial=internal[0]
            parti=Equipartition.particion_vertices(external,intermediate,internal)
            valor=Equipartition.optimizador(poligono,number_of_regions,parti[1],parti[2],1,1) + kappa*(sum(parti[1])-area_del_poligono)**2
            perim=parti[2]
            are=parti[1]
            error_total_normalizado=sum((are/(area_del_poligono/number_of_regions)-1)**2)+sum((perim/perim.mean()-1)**2)
            error_convexidad = sum(parti[1])-area_del_poligono
            error_convexidad_normalizada = sum(parti[1])/area_del_poligono - 1

            jacobian=Equipartition.Jac_AP(poligono,number_of_regions,external,intermediate,internal,10**-5)
            AP_value=Equipartition.AP(poligono,number_of_regions,external,intermediate,internal)

            resp = np.linalg.lstsq(jacobian[0],-AP_value,rcond=None)
            factor0=10**0
            eta=10**-4
            t=0

            while (t < iteraciones_Newton) and ((error_total_normalizado >10**-16)):# or (error_convexidad >10**-5)): #and (distancia > 10**-6 ): 
                rutina_factor_inicial_chico = True
                rutina_puntos_en_el_interior = True  #
                rutina_achicamiento_vector = False   #Rutina de Mayita

                
                if (rutina_factor_inicial_chico == True):
                    if (t < 10):
                        factor = factor0*(t+1)/10
                    else :
                        factor = factor0
                else :
                    factor = factor0
                
                jacobian = Equipartition.Jac_AP(poligono,number_of_regions,external,intermediate,internal,10**-4)
                AP_value = Equipartition.AP(poligono,number_of_regions,external,intermediate,internal)
                error = sum(AP_value[:,0]**2)
                X = np.array([internal[0].flatten()]).transpose() # coordenadas de los internal points vuelto columna
                n = np.size(X)
                m = np.size(AP_value)
                p= np.shape(intermediate[0])[0]
                resp = np.linalg.lstsq(jacobian[0],-AP_value,rcond=None)

                # Routine that checks if the points are inside the polygon 

                if (rutina_puntos_en_el_interior == True) :
                    puntos_intermedios = intermediate[0]+factor*(np.append(resp[0][n:n+p,:],resp[0][n:n+p,:],axis=1)*jacobian[1])
                    puntos_internos = internal[0]+factor*(resp[0][0:n].transpose().reshape((int(np.shape(X)[0]/2),2)))
                    #Si no están los puntos dentro de la figura, achicar el factor 1/2
                    while (np.prod(poligon_path.contains_points(puntos_intermedios, radius=10**-4))*np.prod(poligon_path.contains_points(puntos_internos, radius=10**-4)) == 0) :
                        factor = factor/2
                        puntos_intermedios = intermediate[0]+factor*(np.append(resp[0][n:n+p,:],resp[0][n:n+p,:],axis=1)*jacobian[1])
                        puntos_internos = internal[0]+factor*(resp[0][0:n].transpose().reshape((int(np.shape(X)[0]/2),2)))
                    
                    # The factor used is the one that keeps the points inside the polygon

                intermediate_new=[]
                intermediate_new.append(intermediate[0]+factor*(np.append(resp[0][n:n+p,:],resp[0][n:n+p,:],axis=1)*jacobian[1]))
                intermediate_new.append(intermediate[1])
                intermediate_new.append(intermediate[2])
                internal_new=[]
                internal_new.append(internal[0]+factor*(resp[0][0:n].transpose().reshape((int(np.shape(X)[0]/2),2))))
                internal_new.append(internal[1])
                AP_value_new = Equipartition.AP(poligono,number_of_regions,external,intermediate_new,internal_new)

                if (rutina_achicamiento_vector == True): 
                    while (sum(AP_value_new[:,0]**2) > (1-2*factor*eta)*error) :
                        factor = factor/2
                        intermediate_new=[]
                        intermediate_new.append(intermediate[0]+factor*(np.append(resp[0][n:n+p,:],resp[0][n:n+p,:],axis=1)*jacobian[1]))
                        intermediate_new.append(intermediate[1])
                        intermediate_new.append(intermediate[2])
                        internal_new=[]
                        internal_new.append(internal[0]+factor*(resp[0][0:n].transpose().reshape((int(np.shape(X)[0]/2),2))))
                        internal_new.append(internal[1])
                        AP_value_new = Equipartition.AP(external,intermediate_new,internal_new)

                parti = Equipartition.particion_vertices(external,intermediate_new,internal_new)
                error_convexidad = sum(parti[1])-area_del_poligono
                error_convexidad_normalizada = sum(parti[1])/area_del_poligono - 1
                
                error_total_normalizado=sum((parti[1]/(area_del_poligono/number_of_regions)-1)**2)+sum((parti[2]/parti[2].mean()-1)**2)   
                internal=internal_new
                intermediate=intermediate_new                                                                                                              
                time.sleep(0.01)
                progress = (t + 1) / iteraciones_Newton * 100
                print("\rIterations: |{0:50s}| ".format('█' * int(progress / 2), progress), end="")
                print("\r Iterations: |{0:50s}| ".format('█' * int(progress / 2), progress), end="")

                if (error_convexidad_normalizada > 1) or (factor < 10**-8):
                    break
                t = t+1

            parti=Equipartition.particion_vertices(external,intermediate,internal)
            error_convexidad = sum(parti[1])-area_del_poligono

            intermediate_final=intermediate[0]
            internal_final=internal[0]
            
            distancia_total_final = np.linalg.norm(np.vstack((intermediate_final-intermediate_initial, internal_final-internal_initial)))
            if len(internal_final)>0 :
                distancia_internal_max =((sum((internal_final-internal_initial).transpose()**2))**(1/2)).max()
            else:
                distancia_internal_max = 0   
            distancia_intermediate_max =((sum((intermediate_final-intermediate_initial).transpose()**2))**(1/2)).max()



            perim=parti[2]
            are=parti[1]
            error_total=sum((are-(area_del_poligono/number_of_regions))**2)+sum((perim-perim.mean())**2)
            error_total_normalizado=sum((are/(area_del_poligono/number_of_regions)-1)**2)+sum((perim/perim.mean()-1)**2)

            if ((error_total_normalizado <= 10**-16)):# and (error_convexidad <= 10**-5)):
                alcanzo_resultado=True
            print("Repetition {}, Figure {}, |%F^c|: Total Error={},Convexity Error ={}".format(repeticion,figura,error_total,error_convexidad))
            repeticion=repeticion+1
                


        all_points = np.vstack((np.vstack((external[0],intermediate[0])),internal[0]))

        #------------------------------------------------
        #PANEL DE GRÁFICAS

        all_points = np.vstack((np.vstack((external[0],intermediate[0])),internal[0]))
        all_points_initial = np.vstack((np.vstack((external[0],intermediate_initial)),internal_initial))
        fig, axs = plt.subplots(2, 3,  figsize=(12,8))
        axs[0, 0].set_title('Initial Polygon', fontsize='10')
        axs[0, 1].set_title('Rescaled Polygon',fontsize='10')
        axs[0, 2].set_title('Random Voronoi Partition in Rescaled Polygon',fontsize='10')
        axs[1, 0].set_title('Cetroidal Voronoi Partition in Rescaled Polygon',fontsize='10')
        axs[1, 1].set_title('Cetroidal Voronoi Partition in Initial Polygon',fontsize='10')
        axs[1, 2].set_title(r'Fair Partition: |%$F^c|^2$={}'.format("%.3g" % error_total),fontsize='10')  #"%.3g" % (redondea) :)

        axs[0, 0].plot(poligono[:,0], poligono[:,1], 'k')
        for simplex in poligon.simplices:
            axs[0, 0].plot(poligon.points[simplex, 0], poligon.points[simplex, 1], 'k')
        axs[0, 0].plot(poligon.points[poligon.vertices,0], poligon.points[poligon.vertices,1], '-k')
        axs[0, 0].plot(poligon.points[poligon.vertices[0],0], poligon.points[poligon.vertices[0],1], '-k')


        nuevo_poli=ConvexHull(nuevo_poligono)
        axs[0, 1].plot(nuevo_poligono[:,0], nuevo_poligono[:,1], 'k')
        for simplex in nuevo_poli.simplices:
            axs[0, 1].plot(nuevo_poli.points[simplex, 0], nuevo_poli.points[simplex, 1], '-k')
        axs[0, 1].plot(nuevo_poli.points[nuevo_poli.vertices,0], nuevo_poli.points[nuevo_poli.vertices,1], '-k')
        axs[0, 1].plot(nuevo_poli.points[nuevo_poli.vertices[0],0], nuevo_poli.points[nuevo_poli.vertices[0],1], '-k')

        regions=Equipartition.particion(nuevo_poligono,sitios)[0]
        regions2=Equipartition.particion(nuevo_poligono,coordenadas)[0]
        for n in range(0,len(regions),1):
            for i in range(-1,np.shape(regions[n])[0]-1,1):
                axs[0, 2].plot([regions[n][i][0],regions[n][i+1][0]],[regions[n][i][1],regions[n][i+1][1]],c='k')    

        for n in range(0,len(regions2),1):
            for i in range(-1,np.shape(regions2[n])[0]-1,1):
                axs[1, 0].plot([regions2[n][i][0],regions2[n][i+1][0]],[regions2[n][i][1],regions2[n][i+1][1]],c='k')


        for n in range(0,len(indices_regiones),1):
            for i in range(0,len(indices_regiones[n]),1):
                axs[1, 1].plot([all_points_initial[indices_regiones[n],0][i-1],all_points_initial[indices_regiones[n],0][i]],[all_points_initial[indices_regiones[n],1][i-1],all_points_initial[indices_regiones[n],1][i]], '-k')
                axs[1, 2].plot([all_points[indices_regiones[n],0][i-1],all_points[indices_regiones[n],0][i]],[all_points[indices_regiones[n],1][i-1],all_points[indices_regiones[n],1][i]], '-k')
                centro = Equipartition.centroide(all_points[indices_regiones[n]])
        plt.show()
        if Regions==True:
            print("Coordinates of the regions:{}".format(parti[0]))
        if Areas==True:
            print("Areas:{}".format(parti[1]))
        if Perimeters==True:
            print("Perimeters:{}".format(parti[2]))


