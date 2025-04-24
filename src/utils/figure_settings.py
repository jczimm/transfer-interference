import matplotlib as mpl

med_fontsize = 15
schedule_colours = ['#E69F00', '#56B4E9', '#D9544D']
condition_order = ['same', 'near', 'far']
task_colours = ['#b08cca','#aad39b']
cm_conv = 1 / 2.54  

# Matplotlib style settings
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.25
mpl.rcParams['ytick.minor.width'] = 0.25
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.frameon'] = False
