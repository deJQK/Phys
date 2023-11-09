## Inspired by Ziming Liu @MIT

import os
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
# import matplotlib.collections as mcoll
# import matplotlib.path as mpath

def prob_std_norm(x, sigma=1):
    return np.exp(-0.5 * (x / sigma)**2) / np.sqrt(2 * np.pi) / sigma

# def cmap()

Nsigmas = 100

cmap = clr.LinearSegmentedColormap.from_list('custom cmap', 
                                             [(0,    '#ff0000'),
                                              (1,    '#0000ff')],
                                             N=Nsigmas)
# plt.rcParams['axes.prop_cycle'] = plt.cycler('color', cmap(np.linspace(0, 1, cmap.N)))


# def colorline(
#     x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
#         linewidth=3, alpha=1.0):
#     """
#     http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
#     http://matplotlib.org/examples/pylab_examples/multicolored_line.html
#     Plot a colored line with coordinates x and y
#     Optionally specify colors in the array z
#     Optionally specify a colormap, a norm function and a line width
#     """

#     # Default colors equally spaced on [0,1]:
#     if z is None:
#         z = np.linspace(0.0, 1.0, len(x))

#     # Special case if a single number:
#     if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
#         z = np.array([z])

#     z = np.asarray(z)

#     segments = make_segments(x, y)
#     lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
#                               linewidth=linewidth, alpha=alpha)

#     ax = plt.gca()
#     ax.add_collection(lc)

#     return lc


# def make_segments(x, y):
#     """
#     Create list of line segments from x and y coordinates, in the correct format
#     for LineCollection: an array of the form numlines x (points per line) x 2 (x
#     and y) array
#     """

#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     return segments


def main():
    problem_def = ['gmm', 'data'][-1]
    res_dir = f'./results/{problem_def}'
    os.makedirs(res_dir, exist_ok=True)
    
    if problem_def == 'gmm':
        num_modes = 3
        eps = 0 #1e-20
        alphas = [2] * num_modes
        probs = np.random.dirichlet(alphas) #[0.1, 0.35, 0.55]
        print('Normalization: ', np.sum(probs))
        probs = probs / np.sum(probs)
        print('Pi: ', probs)
        # centers = np.random.rand(num_modes) * 2 - 1
        centers = np.array([-0.8, 0.1, 0.6])
        print('Loc: ', centers)
        
        ## pdf
        pdf_comps = lambda x, sigma: probs[None, None, :] * prob_std_norm(x[:, None, None] - centers[None, None, :], sigma[None, :, None])
        pdf_tots = lambda x, sigma: np.sum(pdf_comps(x, sigma), axis=-1)
        
        ## mean function
        mean_func = lambda x, sigma: np.sum(pdf_comps(x, sigma) * centers[None, None, :], axis=-1) / pdf_tots(x, sigma)
        sigmas = np.logspace(-1.3, 0, Nsigmas)
        norm = mlp.colors.LogNorm(vmin=min(sigmas), vmax=max(sigmas))
        Nxs = 1000
        xs = np.linspace(-2, 2, Nxs)
        means = mean_func(xs, sigmas)
        # means = np.nan_to_num(means, nan=10)
        pdfs = pdf_tots(xs, sigmas)
        renormalized = True
        if renormalized:
            pdfs = pdfs / np.max(pdfs, axis=0, keepdims=True)
        
        ## var function
        mean_sq_func = lambda x, sigma: np.sum(pdf_comps(x, sigma) * centers[None, None, :] ** 2, axis=-1) / (eps + pdf_tots(x, sigma))
        var_func = lambda x, sigma: mean_sq_func(x, sigma) - mean_func(x, sigma) ** 2
                
        ## PDF
        plt.figure(figsize=(8, 6))
        for idx, sigma in enumerate(sigmas):
            plt.plot(xs, pdfs[:, idx], c=cmap(idx / Nsigmas), alpha=0.2, lw=0.5, ls='-', label=r'$\sigma$'+f'={sigma:.2f}')
        # plt.legend(loc='upper right', ncol=3)
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        # cbar.ax.set_yticks(np.linspace(0, 1, 6))
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.savefig(f'{res_dir}/pdf_{num_modes}modes.png', dpi=300)
        
        ## Center function
        plt.figure(figsize=(8, 6))
        for idx, sigma in enumerate(sigmas):
            plt.plot(xs, means[:, idx], c=cmap(idx / Nsigmas), alpha=0.2, lw=0.5, label=r'$\sigma$'+f'={sigma:.2f}')
        # plt.legend(loc='upper left', ncol=3)
        plt.plot(xs, xs, c='k', ls='--', lw=0.5)
        # plt.gca().axhline(y=0, color='k')
        # plt.gca().axvline(x=0, color='k')
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        # cbar.ax.set_yticks(np.linspace(0, 1, 6))
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig(f'{res_dir}/center_func_{num_modes}modes.png', dpi=300)
        
        ## Local minimum vs sigma
        # threshold = 1e-5
        distance = np.abs(means - xs[:, None]) # Nxs x Nsigmas
        local_minimum_idx = np.argpartition(distance, num_modes, axis=0)
        distance_minimum = np.take_along_axis(distance, local_minimum_idx, axis=0)
        print(distance_minimum)
        distance_minimum_top_num_modes = distance_minimum[:2 * num_modes - 1]
        print(distance_minimum_top_num_modes)
        # print(np.max(distance_minimum_top_num_modes, axis=0) / np.min(distance_minimum_top_num_modes, axis=0))
        # print(local_minimum_idx.shape)
        local_minimum = np.take_along_axis(means, local_minimum_idx, axis=0)[:2 * num_modes - 1]
        plt.figure(figsize=(8, 6))
        for loc_mins in local_minimum:
            plt.scatter(sigmas, loc_mins,
                        s=1.5,
                        c=cmap(np.arange(Nsigmas) / Nsigmas))
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.savefig(f'{res_dir}/local_min_vs_sigma_{num_modes}modes.png', dpi=300)
        
        ##  vs sigma
    elif problem_def == 'data':
        num_data = 10 #1 #8 #10
        seed = np.random.randint(0, 10000000)
        seed = 1631487 #3548099
        print('seed: ', seed)
        np.random.seed(seed) #39, 76, 5682, 8284, 6221206, 3548099, 1631487
        # print('seed: ', np.random.get_state()[1][0])
        clean_data = np.random.randn(num_data) * 5
        # clean_data = np.array([4.897, -1.355, 7.130, 2.949, 0.400, -0.867, 0.785, 1.148])
        # print('seed: ', np.random.get_state()[1][0])
        probs = np.ones_like(clean_data) / num_data
        print('Clean data: ', clean_data)
        clean_data_mean = np.mean(clean_data)
        clean_data_mean_square = np.mean(clean_data ** 2)
        clean_data_std = (clean_data_mean_square - clean_data_mean ** 2) ** 0.5
        # [ 4.89655056, -1.35479264, 7.1295406, 2.9486758, 0.39969913, -0.86692981, 0.78505673, 1.14766747]
        
        ## pdf
        pdf_comps = lambda x, sigma: probs[None, None, :] * prob_std_norm(x[:, None, None] - clean_data[None, None, :], sigma[None, :, None])
        pdf_tots = lambda x, sigma: np.sum(pdf_comps(x, sigma), axis=-1)
        
        ## mean function
        mean_func = lambda x, sigma: np.sum(pdf_comps(x, sigma) * clean_data[None, None, :], axis=-1) / pdf_tots(x, sigma)
        sigmas = np.logspace(-2, 1, Nsigmas)
        norm = mlp.colors.LogNorm(vmin=min(sigmas), vmax=max(sigmas))
        Nxs = 1000
        xs = np.linspace(np.min(clean_data) - 2, np.max(clean_data) + 2, Nxs)
        means = mean_func(xs, sigmas)
        # means = np.nan_to_num(means, nan=10)
        pdfs = pdf_tots(xs, sigmas)
        renormalized = True
        if renormalized:
            pdfs = pdfs / np.max(pdfs, axis=0, keepdims=True)
        
        ## var function
        mean_sq_func = lambda x, sigma: np.sum(pdf_comps(x, sigma) * clean_data[None, None, :] ** 2, axis=-1) / (eps + pdf_tots(x, sigma))
        var_func = lambda x, sigma: mean_sq_func(x, sigma) - mean_func(x, sigma) ** 2
        
        ## PDF
        plt.figure(figsize=(8, 6))
        plt.vlines(clean_data, 0, 1, lw=1, ls=':', colors='k')
        for idx, sigma in enumerate(sigmas):
            plt.plot(xs, pdfs[:, idx], c=cmap(idx / Nsigmas), alpha=0.2, lw=0.5, ls='-', label=r'$\sigma$'+f'={sigma:.2f}')
        # plt.legend(loc='upper right', ncol=3)
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        # cbar.ax.set_yticks(np.linspace(0, 1, 6))
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.savefig(f'{res_dir}/pdf_{num_data}data.png', dpi=300)
        
        ## Free Energy
        plt.figure(figsize=(8, 6))
        plt.vlines(clean_data, 0, 1, lw=1, ls=':', colors='k')
        for idx, sigma in enumerate(sigmas):
            plt.plot(xs, -sigma**2 * np.log(pdfs[:, idx]), c=cmap(idx / Nsigmas), alpha=0.2, lw=0.5, ls='-', label=r'$\sigma$'+f'={sigma:.2f}')
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        # cbar.ax.set_yticks(np.linspace(0, 1, 6))
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.savefig(f'{res_dir}/free_energy_{num_data}data.png', dpi=300)
        
        ## Center function
        plt.figure(figsize=(8, 6))
        for idx, sigma in enumerate(sigmas):
            plt.plot(xs, means[:, idx], c=cmap(idx / Nsigmas), alpha=0.2, lw=0.5, label=r'$\sigma$'+f'={sigma:.2f}')
        # plt.legend(loc='upper left', ncol=3)
        plt.plot(xs, xs, c='k', ls='--', lw=0.5)
        # plt.gca().axhline(y=0, color='k')
        # plt.gca().axvline(x=0, color='k')
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        # cbar.ax.set_yticks(np.linspace(0, 1, 6))
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        plt.savefig(f'{res_dir}/center_func_{num_data}data.png', dpi=300)
        
        ## Local minimum vs sigma
        # threshold = 1e-5
        distance = np.abs(means - xs[:, None]) # Nxs x Nsigmas
        local_minimum_idx = np.argpartition(distance, num_data, axis=0)
        distance_minimum = np.take_along_axis(distance, local_minimum_idx, axis=0)
        # print(distance_minimum)
        distance_minimum_top_num_data = distance_minimum[:2 * num_data - 1]
        # print(distance_minimum_top_num_data)
        # print(np.max(distance_minimum_top_num_data, axis=0) / np.min(distance_minimum_top_num_data, axis=0))
        # print(local_minimum_idx.shape)
        local_minimum = np.take_along_axis(means, local_minimum_idx, axis=0)[:2 * num_data - 1]
        plt.figure(figsize=(8, 6))
        for loc_mins in local_minimum:
            plt.scatter(sigmas, loc_mins,
                        s=1.5,
                        c=cmap(np.arange(Nsigmas) / Nsigmas))
        cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation='vertical',
                                label=r'$\sigma$')
        cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.savefig(f'{res_dir}/local_min_vs_sigma_{num_data}data.png', dpi=300)
        
        ##  vs sigma
        ## TODO
        
        ## ODE Dynamics
        num_traj = 80
        num_steps = 64 #128
        # sigmas_sim = np.logspace(np.log10(sigmas[0]), np.log10(sigmas[-1]), num_steps)
        sigmas_sim = np.linspace(sigmas[0], sigmas[-1], num_steps)
        xs = [np.random.randn(num_traj) * sigmas_sim[-1]]
        data_mean = np.mean(clean_data)
        # step_rate = [10] * 4 + [1] * (num_steps - 1 - 4)
        step_rate = np.ones(num_steps - 1)
        for sigma_idx in range(num_steps - 1):
            curr_x = xs[-1]
            curr_sigma = sigmas_sim[[-1-sigma_idx]]
            prev_sigma = sigmas_sim[[-2-sigma_idx]]
            dsigma = (prev_sigma - curr_sigma)[0]
            dx_dsigma = 1 / curr_sigma[0] * (curr_x - mean_func(curr_x, curr_sigma)[:, 0])
            dx = dx_dsigma * dsigma * step_rate[sigma_idx]
            dx = np.nan_to_num(dx, 0)
            prev_x = curr_x + dx
            xs.append(prev_x)
        all_paths = np.stack(xs, axis=0) # shape: [num_steps - 1, num_traj]
        plt.figure(figsize=(8, 6))
        for traj_idx in range(num_traj):
            plt.plot(all_paths[:, traj_idx], sigmas_sim[::-1], alpha=0.6) #, c=cmap(traj_idx / num_traj))
            # plt.scatter(all_paths[:4, traj_idx], sigmas_sim[-4::-1], m='o', s=2)
        for loc_mins in local_minimum:
            plt.scatter(loc_mins, sigmas,
                        s=1.5,
                        c=cmap(np.arange(Nsigmas) / Nsigmas))
        plt.scatter(all_paths[0, :], np.ones(num_traj) * sigmas_sim[-1], c='r', s=80)
        plt.scatter(all_paths[-1, :], np.ones(num_traj) * sigmas_sim[0], c='b', s=80)
        # print(all_paths[-1, :])
        xmin = min(np.min(all_paths), np.min(clean_data) - 2)
        xmax = max(np.max(all_paths), np.max(clean_data) + 2)
        plt.hlines([clean_data_std], xmin, xmax, lw=2, ls='--', colors='m')
        plt.vlines(clean_data, sigmas_sim[0], sigmas_sim[-1], lw=1, ls=':', colors='k')
        plt.vlines([data_mean], sigmas_sim[0], sigmas_sim[-1], lw=2, ls='--', colors='g')
        ## Meshgrid plot
        all_x = np.linspace(xmin, xmax, 100)
        all_sigma = np.linspace(np.min(sigmas_sim), np.max(sigmas_sim), 50)
        xx, yy = np.meshgrid(all_x, all_sigma)
        # print(xx.shape, yy.shape)
        # print(all_x)
        # print(mean_func(all_x, all_sigma))
        moving_dir = (all_x[:, None] < mean_func(all_x, all_sigma)) ## blue (0) for go left, red (1) for go right (as sigma decreasing)
        # moving_vel = (all_x[:, None] - mean_func(all_x, all_sigma)) ## blue (0) for go left, red (1) for go right (as sigma decreasing)
        # print(moving_dir)
        # moving_dir = moving_dir.reshape(xx.shape)
        # print(np.unique(moving_dir.ravel()))
        cmap_for_meshgrid = clr.LinearSegmentedColormap.from_list('custom cmap',
                                                                  [(0, 'b'), (1, 'r')],
                                                                  N=2,
                                                                #   N=200,
                                                                  )
        plt.contourf(xx, yy, moving_dir.T, alpha=0.2, cmap=cmap_for_meshgrid, antialiased=True)
        # plt.contourf(xx, yy, moving_vel.T, alpha=0.2, cmap=cmap_for_meshgrid, antialiased=True)
        # plt.arrow(0.4, 1.2, -0.1, 0,
        #           facecolor='b', edgecolor='none', alpha=0.2,
        #           width=0.01,
        #           transform=plt.gca().transAxes)
        # plt.arrow(0.6, 1.2, 0.1, 0,
        #           facecolor='r', edgecolor='none', alpha=0.2,
        #           width=0.01,
        #           transform=plt.gca().transAxes)
        plt.gca().annotate('', xy=(0.3, 1.05), xycoords='axes fraction',
                           xytext=(0.4, 1.05),
                           arrowprops=dict(arrowstyle='->', linewidth=3, edgecolor='b', alpha=0.2))
        plt.gca().annotate('', xy=(0.7, 1.05), xycoords='axes fraction',
                           xytext=(0.6, 1.05),
                           arrowprops=dict(arrowstyle='->', linewidth=3, edgecolor='r', alpha=0.2))
        plt.xlabel('x')
        plt.ylabel(r'$\sigma$', rotation=0)
        # plt.xlim(np.min(clean_data) - 2, np.max(clean_data) + 2)
        # plt.ylim(0.0, 4)
        plt.ylim(bottom=0)
        # cbar = plt.gcf().colorbar(mlp.cm.ScalarMappable(norm=norm, cmap=cmap),
        #                         orientation='vertical',
        #                         label=r'$\sigma$')
        # cbar.ax.set_ylabel(r'$\sigma$', rotation=0)
        plt.savefig(f'{res_dir}/ODE_trajectory_{num_data}data.png', dpi=300)
    else:
        raise NotImplementedError
    
if __name__ == '__main__':
    main()