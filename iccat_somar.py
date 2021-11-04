#! -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


### Open 1x1 and 5x5 grids
''' with open('iccat_t2ce_somar_1x1.csv') as csvfile:
    f1 = list(csv.reader(csvfile))

with open('iccat_t2ce_somar_5x5.csv') as csvfile:
    f5 = list(csv.reader(csvfile)) '''

### Open allgrids
'''with open('iccat_t2ce_somar_allgrids.csv') as csvfile:
    fa = list(csv.reader(csvfile))'''

### Open South Atlantic
with open('iccat_t2ce_SAt_allgr_3nov21.csv') as csvfile:
    f = list(csv.reader(csvfile))

    
### Pandas dataframe
headers = f[0]
df = pd.DataFrame(f[1:], columns=headers)


species_list = ['BFT', 'ALB', 'YFT', 'BET', 'SKJ', 'SWO', 'BUM', 'SAI', 'SPF', 'WHM', 'BLF', 'BLT', 'BON', 'BOP', 'BRS', 'CER', 'FRI', 'KGM', 'LTA', 'MAW', 'SLT', 'SSM', 'WAH', 'DOL', 'BIL', 'BLM', 'MSP', 'MLS', 'RSP', 'SBF', 'oTun', 'BSH', 'POR', 'SMA', 'oSks']

for var in ['Eff1', 'Eff2', 'BFT', 'ALB', 'YFT', 'BET', 'SKJ', 'SWO', 'BUM', 'SAI', 'SPF', 'WHM', 'BLF', 'BLT', 'BON', 'BOP', 'BRS', 'CER', 'FRI', 'KGM', 'LTA', 'MAW', 'SLT', 'SSM', 'WAH', 'DOL', 'BIL', 'BLM', 'MSP', 'MLS', 'RSP', 'SBF', 'oTun', 'BSH', 'POR', 'SMA', 'oSks']:
    df[var] = pd.to_numeric(df[var], errors='coerce')

df = df.astype({'StrataID': 'int64', 'DSetID': 'int64', 'FleetID': 'string', 'GearGrpCode': 'string', 'GearCode': 'string', 'FileTypeCode': 'string', 'YearC': 'int64', 'TimePeriodID': 'int64', 'SquareTypeCode': 'string', 'QuadID': 'int64', 'Lat': 'float64', 'Lon': 'float64', 'SchoolTypeCode': 'string', 'Eff1': 'float64', 'Eff1Type': 'string', 'Eff2': 'float64', 'Eff2Type': 'string', 'DSetTypeID': 'string', 'CatchUnit': 'string', 'BFT': 'float64', 'ALB': 'float64', 'YFT': 'float64', 'BET': 'float64', 'SKJ': 'float64', 'SWO': 'float64', 'BUM': 'float64', 'SAI': 'float64', 'SPF': 'float64', 'WHM': 'float64', 'BLF': 'float64', 'BLT': 'float64', 'BON': 'float64', 'BOP': 'float64', 'BRS': 'float64', 'CER': 'float64', 'FRI': 'float64', 'KGM': 'float64', 'LTA': 'float64', 'MAW': 'float64', 'SLT': 'float64', 'SSM': 'float64', 'WAH': 'float64', 'DOL': 'float64', 'BIL': 'float64', 'BLM': 'float64', 'MSP': 'float64', 'MLS': 'float64', 'RSP': 'float64', 'SBF': 'float64', 'oTun': 'float64', 'BSH': 'float64', 'POR': 'float64', 'SMA': 'float64', 'oSks': 'float64'})


### Sum of annual captures by species, allgrids
''' sumFS = df.groupby(['CatchUnit', 'YearC'])[species_list].sum()
print(sumFS.dtypes)
#sumFS = sumFS.loc[sumFS['CatchUnit'] == 'kg']

sumFS.to_csv('Annual_catch_ICCAT_SOMAR_allgrids.csv')'''


### Slice into 3 polygons, islands SP AI SH
'''poly_sp_v = [(-37,0), (-28,8), (-15, 3), (-25, -10)]
poly_ai_v = [(-25,-10), (-15, 3), (-3.9, -5), (-16, -20)]
poly_sh_v = [(-16,-20), (-3.9,-5), (3,-15), (-5,-24)]

poly_sp = mpath.Path(poly_sp_v)
poly_ai = mpath.Path(poly_ai_v)
poly_sh = mpath.Path(poly_sh_v)

lon, lat = df['Lon'], df['Lat']
coords = np.array([lon, lat]).T
contains_sp = poly_sp.contains_points(coords)
contains_ai = poly_ai.contains_points(coords)
contains_sh = poly_sh.contains_points(coords)

df_sp = pd.DataFrame(df[contains_sp], columns = fa[0])
df_ai = pd.DataFrame(df[contains_ai], columns = fa[0])
df_sh = pd.DataFrame(df[contains_sh], columns = fa[0])'''


### Sum of annual captures by species, 3 polys SP AI SH
''' sumFS = df_sp.groupby(['CatchUnit', 'YearC'])[species_list].sum()
sumFS.to_csv('Annual_catch_ICCAT_SOMAR_SP.csv')

sumFS = df_ai.groupby(['CatchUnit', 'YearC'])[species_list].sum()
sumFS.to_csv('Annual_catch_ICCAT_SOMAR_AI.csv')

sumFS = df_sh.groupby(['CatchUnit', 'YearC'])[species_list].sum()
sumFS.to_csv('Annual_catch_ICCAT_SOMAR_SH.csv') '''

### Sum of 5 yr capture, StrathE2E
df_5yr = df[df['YearC'] > 2011]   # select years


for var in ['GearGrpCode', 'GearCode', 'SquareTypeCode', 'SchoolTypeCode', 'Eff1Type', 'Eff2Type', 'CatchUnit']:
    plot = df_5yr[var].value_counts().plot.pie(autopct='%.1f%%')
    plt.show()
    plt.clf()

exit()

#df_5yr = df[df['CatchUnit'] == 'kg']


#species = 'oTun'       # select species
#sum_5yr = df_5yr.groupby(['Lon', 'Lat'], as_index=False)[species_list].sum()

Lon = np.arange(-46, 12, 1)   # gridding Lat, Lon and data (C)
Lat = np.arange(-31, 16, 1)


proj = ccrs.PlateCarree()
fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw=dict(projection=proj))

def plot_t2ce(ax, species):
    ax.coastlines()
    ax.set_extent([-45, 10, -30, 15], crs=proj)
    gl = ax.gridlines(alpha=0, draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    sum_5yr = df_5yr.groupby(['Lon', 'Lat'], as_index=False)[species].sum()

    C = np.empty((len(Lon), len(Lat)))
    C[:] = np.nan           # All NaNs matrix, to be filled with data

    for i, lo in enumerate(Lon):
        for j, la in enumerate(Lat):
            catch = sum_5yr[sum_5yr['Lon']==lo+.5]   # gridding data centered on 1x1
            catch = catch[catch['Lat']==la+.5]

            try:
                catch2 = catch.iat[0, 2]
            except IndexError:
                pass
            else:
                catch2 = catch.iat[0, 2]
                if catch2 > 0:
                    C[i, j] = catch2

    print(species, np.nanmin(C), np.nanmax(C))


    #cm = ax.pcolormesh(Lon, Lat, C.T)
    cm = ax.pcolormesh(Lon, Lat, C.T, norm=mcolors.LogNorm(vmin=1e1, vmax=1e7))

    poly_verts = [(-36,0), (-28,9), (3,-15), (-5,-24)]  # plotting SOMAR polygon
    patch_somar = mpatches.Polygon(poly_verts, ec='r', fc='None', alpha=.5)
    ax.add_patch(patch_somar)

    ax.set_title(species)

    return cm

CM = plot_t2ce(ax, 'ALB')  ## species
#plot_t2ce(axs[0,1], 'BET')
#plot_t2ce(axs[1,0], 'YFT')
#CM = plot_t2ce(axs[1,1], 'SKJ')

    
plt.colorbar(CM, ax=axs, label='Total catch 2012-2019 (kg)', shrink=.4, fraction=.05)


plt.savefig('TC_2012-2019_ALLspp_Log.png', dpi=600)


### Plot 3 polygons SP AI SH
''' fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
gl = ax.gridlines(alpha=0, draw_labels=True)
gl.top_labels = False
gl.right_labels = False

patch_sp = mpatches.Polygon(poly_sp_v, ec='b', fc='None')
patch_ai = mpatches.Polygon(poly_ai_v, ec='r', fc='None')
patch_sh = mpatches.Polygon(poly_sh_v, ec='g', fc='None')
ax.add_patch(patch_sp)
ax.add_patch(patch_ai)
ax.add_patch(patch_sh)

ax.scatter(df['Lon'][contains_sp], df['Lat'][contains_sp], c='b', s=5)
ax.scatter(df['Lon'][contains_ai], df['Lat'][contains_ai], c='r', s=5)
ax.scatter(df['Lon'][contains_sh], df['Lat'][contains_sh], c='g', s=5)

plt.savefig('teste2.png', dpi=300) '''


### Plot captures of some species
'''fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
ax.set_extent([-45, 10, -30, 15])
gl = ax.gridlines(alpha=0, draw_labels=True)
gl.top_labels = False
gl.right_labels = False

species = 'SMA'   # select species
square = '5x5'    # select square
factor = .01       # size factor for scatter plot

sum_sp = df_sp[df_sp['SquareTypeCode']==square].groupby(['Lon', 'Lat'], as_index=False)[species].sum()
sum_ai = df_ai[df_ai['SquareTypeCode']==square].groupby(['Lon', 'Lat'], as_index=False)[species].sum()
sum_sh = df_sh[df_sh['SquareTypeCode']==square].groupby(['Lon', 'Lat'], as_index=False)[species].sum()

ax.scatter(sum_sp['Lon'], sum_sp['Lat'], s=sum_sp[species] * factor, alpha=0.5, c='b', label='SPSP')
ax.scatter(sum_ai['Lon'], sum_ai['Lat'], s=sum_ai[species] * factor, alpha=0.5, c='r', label='AI')
ax.scatter(sum_sh['Lon'], sum_sh['Lat'], s=sum_sh[species] * factor, alpha=0.5, c='g', label='SH')

sum_all = pd.concat([sum_sp[species], sum_ai[species], sum_sh[species]], ignore_index=True)
print(sum_all.min(), sum_all.max())
sum_all = np.array(sum_all)

ms1 = 5
#ms1 = np.around(np.min(sum_all[np.nonzero(sum_all)]), 0)  # Legend
#ms2 = np.around(np.median(sum_all), -3)
#ms3 = np.around(np.max(sum_all), -3)   
l1, = plt.plot([],[], 'ob', markersize=ms1, alpha=.5)
l2, = plt.plot([],[], 'or', markersize=ms1, alpha=.5)
l3, = plt.plot([],[], 'og', markersize=ms1, alpha=.5)
labels = ['SPSP', 'AI', 'SH']
leg = plt.legend([l1, l2, l3], labels, ncol=1, frameon=True, title='Total catch 1950-2019 (kg)\nmax '+str(int(sum_all.max())), scatterpoints=1)


ax.set_title(species+' '+square)
plt.savefig('total_catch_'+square+'_'+species+'.png', dpi=300)'''
