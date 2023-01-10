from run_simulation import *

data_path = 'saved_data/'

#conductance with respect to tip position
def G_simple(N, M, version=1):
	wfac = (1 + 2*w_pot/W) #* 100
	r1fac = (1 - w_pot/r1) #* 0
	r2fac = (1 + w_pot/r2) #* 100
	xx, yy, dtrh = get_positions(N, M, r2_over=r2fac, w_over=wfac, r1_under=r1fac)

	Ef = t/2
	Vg = .9*Ef
	Rp = w_pot

	main(xx, yy, dtrh,  Vg, Ef, Rp, f'{data_path}G_{int(N)}x{int(M)}_pos__{int(version)}.mat')
	main(xx, yy, dtrh, -Vg, Ef, Rp, f'{data_path}G_{int(N)}x{int(M)}_neg__{int(version)}.mat')

#same as G_simple, but 1D (y=0)
def G_1D(N, Ef_factor=.5, only_inbound=False, pos=True, neg=True, version=1):
	x = np.linspace(-r2*.75, -r1, N) if only_inbound else np.linspace(-(2*r2 - r1), 0, N)

	Ef = t*Ef_factor
	Vg = .9*Ef
	Rp = w_pot

	if pos: main(x, np.zeros_like(x), np.ones_like(x, dtype=bool),  Vg, Ef, Rp, f'{data_path}G_1D_{int(N)}_pos__{int(version)}.mat')
	if neg: main(x, np.zeros_like(x), np.ones_like(x, dtype=bool), -Vg, Ef, Rp, f'{data_path}G_1D_{int(N)}_neg__{int(version)}.mat')

#same as G_simple, but with a more precise mesh
def G_advanced(N, M, S, T, Ef_factor=.5, version=1):
	wfac = (1 + 2*w_pot/W) #* 100
	r1fac = (1 - w_pot/r1) #* 0
	r2fac = (1 + w_pot/r2) #* 100
	xx, yy, dtrh = get_positions(N, M, r2_over=r2fac, w_over=wfac, r1_under=r1fac)
	dtrh[(xx <= -r1) & (xx >= -r2) & (yy <= W/2)] = False

	Ef = Ef_factor*t
	Vg = .9*Ef
	Rp = w_pot

	main(xx, yy, dtrh, -Vg, Ef, Rp, f'{data_path}G1_{int(N)}x{int(M)}_neg__{int(version)}.mat')

	#precise mesh
	xx, yy = np.meshgrid(np.linspace(-r2, -r1, S), np.linspace(0, W/2, T))
	main(xx, yy, np.ones_like(xx, dtype=bool), -Vg, Ef, Rp, f'{data_path}G2_{S}x{T}_neg__{int(version)}.mat')

#conductance with respect to x-coordinate of tip and amplitude of phi_p
def Vg_simple(N, M, version=1):
	Ef = t/2
	Vg = np.linspace(-2, 2, M)*Ef
	Rp = w_pot

	xx, Vg = np.meshgrid(np.linspace(-(2*r2 - r1), 0, N), Vg)
	yy = np.zeros_like(xx)

	dtrh = np.ones_like(xx, dtype=bool)
	main(xx, yy, dtrh,  Vg, Ef, Rp, f'{data_path}Vg_{int(M)}_{int(N)}x1__{int(version)}.mat')

#conductance with respect to x-coordinate of tip and FWHM of phi_p
def Rp_simple(N, M, version=1):
	Ef = t/2
	Vg = Ef*.9
	Rp = np.linspace(10e-9, 250e-9, M)

	xx, Rp = np.meshgrid(np.linspace(-(2*r2 - r1), 0, N), Rp)
	yy = np.zeros_like(xx)

	dtrh = np.ones_like(xx, dtype=bool)
	main(xx, yy, dtrh,  Vg, Ef, Rp, f'{data_path}Rp_{int(M)}_{int(N)}x1_pos__{int(version)}.mat')
	main(xx, yy, dtrh, -Vg, Ef, Rp, f'{data_path}Rp_{int(M)}_{int(N)}x1_neg__{int(version)}.mat')

#conductance with respect to x-coordinate of tip and value of E_F
def Ef_simple(N, M, version=1):
	Ef = np.linspace(.2, 2, M) * t/2
	Rp = w_pot

	xx, Ef = np.meshgrid(np.linspace(-(2*r2 - r1), 0, N), Ef)
	yy = np.zeros_like(xx)
	Vg = .9*Ef

	dtrh = np.ones_like(xx, dtype=bool)
	main(xx, yy, dtrh,  Vg, Ef, Rp, f'{data_path}Ef_{int(M)}_{int(N)}x1_pos__{int(version)}.mat')
	main(xx, yy, dtrh, -Vg, Ef, Rp, f'{data_path}Ef_{int(M)}_{int(N)}x1_neg__{int(version)}.mat')

#variations of conductance with respect amplitude of phi_p (while keeping Vg/Ef = .9)
def dG_simple(N, M, version=1):
	Ef = t/2 if M == 1 else np.linspace(.2, 2, M) * t/2
	Vg = np.linspace(0, 2, N)
	Rp = w_pot

	Vg, Ef = np.meshgrid(Vg, Ef)
	Vg = Vg * Ef
	xx = np.ones_like(Vg) * (-1) * r1
	yy = np.zeros_like(xx)

	dtrh = np.ones_like(xx, dtype=bool)
	main(xx, yy, dtrh,  Vg, Ef, Rp, f'{data_path}dG_{int(N)}_avg_{int(M)}_pos__{int(version)}.mat')
	main(xx, yy, dtrh, -Vg, Ef, Rp, f'{data_path}dG_{int(N)}_avg_{int(M)}_neg__{int(version)}.mat')

#variations of conductance with respect to Vg, Ef ratio
def dG_Rp(N, M, Rps=None, version=1):
	Ef = t/2 if M == 1 else np.linspace(.2, 2, M) * t/2
	Vg = np.linspace(0, 2, N)

	Vg, Ef = np.meshgrid(Vg, Ef)
	Vg = Vg * Ef
	xx = np.ones_like(Vg) * (-1) * r1
	yy = np.zeros_like(xx)

	dtrh = np.ones_like(xx, dtype=bool)

	for Rp in np.linspace(130, 200, 8) if Rps is None else Rps:
		main(xx, yy, dtrh,  Vg, Ef, Rp*1e-9, f'{data_path}dG_{int(N)}_avg_{int(M)}_Rp_{int(Rp)}_pos__{int(version)}.mat')
		main(xx, yy, dtrh, -Vg, Ef, Rp*1e-9, f'{data_path}dG_{int(N)}_avg_{int(M)}_Rp_{int(Rp)}_neg__{int(version)}.mat')
