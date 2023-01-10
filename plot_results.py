from run_simulation import *

data_path = 'saved_data/'
scr_mode = not True

#helper for colorbar plots
def plot_cbar(mesh, top=False, **kwargs):
	fig = plt.gcf()

	#create some space on top or bellow the graphs
	if top:
		fig.subplots_adjust(top=.75)
		cbar_ax = fig.add_axes([.3, .8, .4, .05])

	else:
		fig.subplots_adjust(bottom=0.35)
		cbar_ax = fig.add_axes([.3, .2, .4, .05])

	#plot the colorbar and put the label accordingly
	cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal', **kwargs)
	if top:
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	return cbar


def G_helper(data_mat, G0, draw_boundaries=True):
	xx, yy, truths, data = data_mat['xx'], data_mat['yy'], data_mat['truths'].astype(bool), (data_mat['data'] - G0) / G0; data[~truths] = 0
	N, M = xx.shape

	vmin = 8
	vmax = 13
	lwidth = 3 if scr_mode else 1

	cmap = 'gist_heat'

	#plot (and symetrize) data in the four quadrants
	plt.contourf( xx/r2,              yy/r2,              data,             vmin=vmin, vmax=vmax, levels=100, cmap=cmap)
	plt.contourf(-xx[::-1]/r2,        yy[::-1]/r2,        data[::-1],       vmin=vmin, vmax=vmax, levels=100, cmap=cmap)
	plt.contourf( xx[:, ::-1]/r2,    -yy[:, ::-1]/r2,     data[:, ::-1],    vmin=vmin, vmax=vmax, levels=100, cmap=cmap)
	plt.contourf(-xx[::-1, ::-1]/r2,  -yy[::-1, ::-1]/r2, data[::-1, ::-1], vmin=vmin, vmax=vmax, levels=100, cmap=cmap)

	#colorbar/countourf trick
	mesh = plt.contourf([[100, 101], [100, 101]], [[100, 100], [101, 101]], [[vmin, vmax], [vmax, vmin]], levels=100, cmap=cmap)

	if draw_boundaries:
		t = np.linspace(0, 1, 300)
		ccolor = 'black'
		x, y = np.cos(t*np.pi), np.sin(t*np.pi)
		plt.plot(r1/r2*x, r1/r2*y, color=ccolor, linestyle='dashed', linewidth=lwidth)
		plt.plot(x[y >= W/r2/2], y[y >= W/r2/2], color=ccolor, linestyle='dashed', linewidth=lwidth)
		plt.plot(r1/r2*x, -r1/r2*y, color=ccolor, linestyle='dashed', linewidth=lwidth)
		plt.plot(x[y >= W/r2/2], -y[y >= W/r2/2], color=ccolor, linestyle='dashed', linewidth=lwidth)

		t = (2*t - 1) * xmax/r2
		truths = (t*t + ((.5*W/r2)**2)) > 1
		plt.plot(t[truths & (t > 0)], np.ones_like(t)[truths & (t > 0)]*W/2/r2, color=ccolor, linestyle='dashed', linewidth=lwidth)
		plt.plot(t[truths & (t < 0)], np.ones_like(t)[truths & (t < 0)]*W/2/r2, color=ccolor, linestyle='dashed', linewidth=lwidth)
		plt.plot(t[truths & (t > 0)], -np.ones_like(t)[truths & (t > 0)]*W/2/r2, color=ccolor, linestyle='dashed', linewidth=lwidth)
		plt.plot(t[truths & (t < 0)], -np.ones_like(t)[truths & (t < 0)]*W/2/r2, color=ccolor, linestyle='dashed', linewidth=lwidth)

	plt.xlim((-(2*r2 - r1)/r2, (2*r2 - r1)/r2))
	plt.ylim((-1, 1))
	plt.axis('off')
	return mesh

def G_simple(N, M, version=1):
	plt.figure(figsize=(8, 4))
	plt.subplot(1, 2, 1)
	G_helper(loadmat(f'{data_path}G_{N}x{M}_pos__{version}.mat'))
	plt.subplot(1, 2, 2)
	mesh = G_helper(loadmat(f'{data_path}G_{N}x{M}_neg__{version}.mat'))

	plot_cbar(mesh, label='G ($2e^2 / h$)').ax.set_xticks([9, 12])


def G_advanced(N, M, S, T, version=1):
	plt.figure(figsize=(16, 8) if scr_mode else (8, 4))

	#rough mesh
	plt.subplot(1, 2, 1)
	G_helper(loadmat(f'{data_path}G1_{N}x{M}_pos__{version}.mat'))
	G_helper(loadmat(f'{data_path}G2_{S}x{T}_pos__{version}.mat'))

	#precise mesh
	plt.subplot(1, 2, 2)
	G_helper(loadmat(f'{data_path}G1_{N}x{M}_neg__{version}.mat'), G0)
	mesh = G_helper(loadmat(f'{data_path}G2_{S}x{T}_neg__{version}.mat'), G0, False)

	plot_cbar(mesh, label='G ($2e^2 / h$)').ax.set_xticks([9, 12])


def Vg_simple(N, M, version=1):
	data_mat = loadmat(f'{data_path}Vg_{int(M)}_{int(N)}x1__{int(version)}.mat')

	xx, Vg, Ef, data = data_mat['xx'], data_mat['Vg'], data_mat['Ef'], data_mat['data']
	N, M = xx.shape

	#symetrize data
	newX, newVg, newData = np.empty((N, 2*M)), np.empty((N, 2*M)), np.empty((N, 2*M))
	newX[:, :M]  = xx; newX[:, M:]  = -xx[:, ::-1]
	newVg[:, :M] = Vg; newVg[:, M:] = Vg
	newData[:, :M] = data; newData[:, M:] = data[:, ::-1]

	plt.subplot(1, 2, 1)
	plt.contourf(newX/r2, newVg/Ef, newData, vmin=8, vmax=13, cmap='gist_heat', levels=100)
	plt.vlines([-r1/r2, r1/r2], -2, 2, color='black', linestyle='dashed')
	plt.gca().invert_yaxis()

	plt.yticks([-2, 2])
	plt.xlabel('x (1/$r_2$)')
	plt.ylabel('$\\phi_p$ ($E_F$)')
	plot_cbar(mesh, label='G ($2e^2 / h$)').ax.set_xticks([9, 12])


def Rp_helper(data_mat, invert=False):
	xx, Rp, Ef, data = data_mat['xx'], data_mat['Rp'], data_mat['Ef'], data_mat['data']
	N, M = xx.shape
	newX, newRp, newData = np.empty((N, 2*M)), np.empty((N, 2*M)), np.empty((N, 2*M))
	newX[:, :M]  = xx; newX[:, M:]  = -xx[:, ::-1]
	newRp[:, :M] = Rp; newRp[:, M:] = Rp

	newData[:, :M] = data; newData[:, M:] = data[:, ::-1]
	plt.contourf(newX/r2, newRp*1e9, newData, vmin=8, vmax=13, cmap='gist_heat', levels=100)
	mesh = plt.contourf([[100, 101], [100, 101]], [[100, 100], [101, 101]], [[8, 12.5], [12.5, 8]], vmin=8, vmax=13, levels=100, cmap='gist_heat')

	plt.xlim((np.min(newX)/r2, np.max(newX)/r2))
	plt.ylim((np.min(newRp)*1e9, np.max(newRp)*1e9))
	plt.vlines([-r1/r2, r1/r2], 0, 250, color='black', linestyle='dashed')

	if invert:
		plt.gca().invert_yaxis()
		plt.xlabel('x (1/$r_2$)')
	else:
		plt.xticks([])

	plt.yticks([250])

	return mesh

def Rp_simple(N, M, cbar=True, version=1):
	plt.subplot(45, 2, 23*2)
	plt.plot([-1, 1], [0, 0], color='black', linewidth=3)
	plt.xlim([-1, 1])
	plt.xticks([])
	plt.ylim([-1, 1])
	plt.yticks([0])
	plt.ylabel('$R_p$ (nm)')

	plt.subplot(2, 1, 4)
	Rp_helper(loadmat(f'{data_path}Rp_{int(M)}_{int(N)}x1_pos__{int(version)}.mat'), invert=True)

	plt.subplot(2, 1, 2)
	mesh = Rp_helper(loadmat(f'{data_path}Rp_{int(M)}_{int(N)}x1_neg__{int(version)}.mat'), invert=False)

	plot_cbar(mesh, label='G ($2e^2 / h$)').ax.set_xticks([9, 12])


def Ef_helper(data_mat, top=False, tol=1e-3):
	xx, Ef, data = data_mat['xx'], data_mat['Ef'], data_mat['data']
	G0 = data[:, -1]

	N, M = xx.shape
	newX, newEf, newData = np.empty((N, 2*M)), np.empty((N, 2*M)), np.empty((N, 2*M))
	newX[:, :M]  = xx; newX[:, M:]  = -xx[:, ::-1]
	newEf[:, :M] = Ef; newEf[:, M:] = Ef

	newData[:, :M] = data; newData[:, M:] = data[:, ::-1]
	dG = (newData-G0[:, None])/G0[:, None]

	xtr = np.abs(newX[0, :]) <= r1*.5
	truths = ~np.any(np.abs(dG[:, xtr]) > tol, axis=1)

	plt.contourf(newX[truths]/r2, newEf[truths]/(.5*t), dG[truths], cmap='seismic', vmin=-.4, vmax=.4, levels=100)
	plt.vlines([-r1/r2, r1/r2], .2, 2, color='black', linestyle='dashed')
	mesh = plt.contourf([[100, 101], [100, 101]], [[100, 100], [101, 101]], [[-.4, .4], [.4, -.4]], vmin=-.4, vmax=.4, levels=100, cmap='seismic')

	plt.xlim((np.min(newX)/r2, np.max(newX)/r2))
	plt.ylim((.2, 2))
	plt.yticks([.2, 2])
	plt.ylabel('$E_F$ ($t^*$)')
	if top: plt.xticks([])
	else: plt.xlabel('x (1/$r_2$)')

	return mesh

def Ef_simple(N, M, tol=1e-3, version=1):
	plt.subplot(2, 1, 1)
	mesh = Ef_helper(loadmat(f'{data_path}Ef_{int(M)}_{int(N)}x1_pos__{int(version)}.mat'), top=True, tol=tol)

	plt.subplot(2, 1, 2)
	mesh = Ef_helper(loadmat(f'{data_path}Ef_{int(M)}_{int(N)}x1_neg__{int(version)}.mat'), tol=tol)

	plot_cbar(mesh, top=True, label='$\\Delta$G/G$_0$').ax.set_xticks([-.35, 0, .35])


def dG_helper(data_mat, negative=False, shadow=False):
	Vg, Ef, data = data_mat['Vg'], data_mat['Ef'], data_mat['data']
	N, M = Ef.shape

	G = np.mean(data, axis=0)
	plt.plot((Vg/Ef)[0] *((-1) if negative else 1), ((G - G[0])/G[0]), color='blue' if negative else 'red', alpha=.3 if shadow else 1)

	plt.xlabel('$\\phi^{max}_p/E_F$')
	plt.ylabel('$<\\Delta$G/G$_0$>')
	plt.yticks([-.1, 0, .1, .2])

def dG_simple(N, M, version=1):
	dG_helper(loadmat(f'{data_path}dG_{int(N)}_avg_{int(M)}_pos__{int(version)}.mat'))
	dG_helper(loadmat(f'{data_path}dG_{int(N)}_avg_{int(M)}_neg__{int(version)}.mat'), True)

def dG_Rp(N, M, Rps=None, version=1):
	for Rp in np.linspace(130, 200, 8) if Rps is None else Rps:
		dG_helper(loadmat(f'{data_path}dG_{int(N)}_avg_{int(M)}_Rp_{int(Rp)}_pos__{int(version)}.mat'), False, Rp != 150)
		dG_helper(loadmat(f'{data_path}dG_{int(N)}_avg_{int(M)}_Rp_{int(Rp)}_neg__{int(version)}.mat'), True,  Rp != 150)

	plt.subplots_adjust(bottom=.16, left=.18)

def G_1D_helper(data_mat, negative=False, symetrize=True):
	x, data = data_mat['xx'].flatten(), data_mat['data'].flatten()
	lwidth = 2 if scr_mode else 1
	color = 'blue' if negative else 'red'

	plt.plot(x/r2, data, color=color, linewidth=lwidth)
	if symetrize: plt.plot(-x/r2, data, color=color, linewidth=lwidth)

def G_1D(N, pos=True, neg=True, symetrize=True, version=1):
	if pos: G_1D_helper(loadmat(f'{data_path}G_1D_{int(N)}_pos__{int(version)}.mat'), False, symetrize)
	if neg: G_1D_helper(loadmat(f'{data_path}G_1D_{int(N)}_neg__{int(version)}.mat'), True, symetrize)

	plt.vlines([-r1/r2, r1/r2][:2 if symetrize else 1], 0, 20, linestyle='dashed', color='black', linewidth=2 if scr_mode else 1)
	plt.ylim([6.5, 10.5])
	plt.xlim([-1.3, 1.3 if symetrize else 0])

	plt.yticks([7, 8, 9, 10])
	plt.xlabel('Position (1/$r_2$)')
	plt.ylabel('G ($2e^2/h$)')
