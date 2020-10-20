import healpy as hp
from matplotlib.pyplot import *
import meander

# Where to load the input files from
spt3g_infile = (
    '/sptlocal/user/ngoecknerwald/masks_for_spo_forecasting_figure/spt_2018_mask.fits'
)
bk_infile = '/sptlocal/user/ngoecknerwald/masks_for_spo_forecasting_figure/bk14_mask_cel_n0512.fits'
ba_infile = '/sptlocal/user/ngoecknerwald/masks_for_spo_forecasting_figure/bk18_mask_largefield_cel_n0512.fits'
planck_353 = '/spt/simulation/planck_maps/pr3/353/HFI_SkyMap_353_2048_R3.01_full.fits'

# Below this the apodization mask is said to be zero, for floating point issues
threshold = 1e-10

# Method shamelessly stolen from https://icecube.wisc.edu/~icecube-bootcamp/bootcamp2019/advanced_plotting/examples/gw_skymap/gw_neutrino_skymap.py
def compute_contours(fraction, weight):

    # Compute theta and phi values
    nside = hp.pixelfunc.get_nside(weight)
    sample_points = np.array(hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))).T

    # Cut all of the full sky zero weight pixels to dramatically speed up the contour calculation
    sl = weight > threshold
    weight = weight[sl]
    sample_points = sample_points[sl, :]

    # Compute contours in theta, phi
    pts = meander.compute_contours(
        sample_points, weight, fraction, geodesic='spherical'
    )
    theta_list = pts[0][0][:, 0]
    phi_list = pts[0][0][:, 1]

    ra_list = phi_list * 180 / np.pi
    dec_list = 90 - (theta_list * 180 / np.pi)

    return ra_list, dec_list


def weight_to_fsky(weight_map, mode='signal'):

    assert mode in 'signal noise clem'.split()

    sl = weight_map > threshold

    if mode == 'signal':
        factor = np.mean(weight_map[sl] ** 2) ** 2 / np.mean(weight_map[sl] ** 4)
    elif mode == 'noise':
        factor = np.mean(weight_map[sl]) ** 2 / np.mean(weight_map[sl] ** 2)
    else:
        factor = np.mean(weight_map[sl])

    return factor * float(np.sum(weight_map > threshold)) / float(weight_map.shape[0])


def main():

    # Load the map input files
    spt3g_map = hp.read_map(spt3g_infile)
    spt3g_map[np.logical_not(np.isfinite(spt3g_map))] = 0.0
    spt3g_map /= np.max(spt3g_map)

    bk_map = hp.read_map(bk_infile)
    bk_map[np.logical_not(np.isfinite(bk_map))] = 0.0
    bk_map /= np.max(bk_map)

    ba_map = hp.read_map(ba_infile)
    ba_map[np.logical_not(np.isfinite(ba_map))] = 0.0
    ba_map /= np.max(ba_map)

    planck_map = hp.read_map(planck_353)

    # Put the planck map in celestial coords like everything else
    rotate = hp.Rotator(coord='CG')
    pixnums = np.arange(planck_map.shape[0])
    t, p = hp.pix2ang(hp.npix2nside(planck_map.shape[0]), pixnums)
    trot, prot = rotate(t, p)
    reordering = hp.pixelfunc.ang2pix(hp.npix2nside(planck_map.shape[0]), trot, prot)
    planck_map = planck_map[reordering]

    square_deg_sphere = 4 * np.pi * (180.0 / np.pi) ** 2

    # Now let's compute fsky numbers, all in square degrees
    bk_signal = weight_to_fsky(bk_map, mode='signal') * square_deg_sphere
    bk_noise = weight_to_fsky(bk_map, mode='noise') * square_deg_sphere
    bk_clem = weight_to_fsky(bk_map, mode='clem') * square_deg_sphere
    print(
        'BK apodization mask fsky numbers %s / %s / %s S/N/C'
        % (bk_signal, bk_noise, bk_clem)
    )

    ba_signal = weight_to_fsky(ba_map, mode='signal') * square_deg_sphere
    ba_noise = weight_to_fsky(ba_map, mode='noise') * square_deg_sphere
    ba_clem = weight_to_fsky(ba_map, mode='clem') * square_deg_sphere
    print(
        'BA apodization mask fsky numbers %s / %s / %s S/N/C'
        % (ba_signal, ba_noise, ba_clem)
    )

    spt3g_signal = weight_to_fsky(spt3g_map, mode='signal') * square_deg_sphere
    spt3g_noise = weight_to_fsky(spt3g_map, mode='noise') * square_deg_sphere
    spt3g_clem = weight_to_fsky(spt3g_map, mode='clem') * square_deg_sphere
    print(
        'BA apodization mask fsky numbers %s / %s / %s S/N/C'
        % (spt3g_signal, spt3g_noise, spt3g_clem)
    )

    # Now let's turn this fsky into fractions of the apodization mask using the signal definition
    bk_thresh = (np.sort(bk_map)[::-1])[
        int(bk_clem / square_deg_sphere * float(bk_map.shape[0]))
    ]
    ba_thresh = (np.sort(ba_map)[::-1])[
        int(ba_clem / square_deg_sphere * float(ba_map.shape[0]))
    ]
    spt3g_thresh = (np.sort(spt3g_map)[::-1])[
        int(spt3g_signal / square_deg_sphere * float(spt3g_map.shape[0]))
    ]
    print('Weight thresholds %f %f %f' % (bk_thresh, ba_thresh, spt3g_thresh))

    # Now, let's plot things
    hp.mollview(-1.0 * planck_map, norm='hist', rot=None, cmap=cm.gray)
    # Compute contours
    spt3g_boundary = compute_contours([spt3g_thresh], spt3g_map)
    bk_boundary = compute_contours([bk_thresh], bk_map)
    ba_boundary = compute_contours([ba_thresh], ba_map)

    np.savetxt('bicep_array_footprint.txt', np.transpose([ba_boundary[0], ba_boundary[1]]))

    # Plot contours
    hp.projplot(
        bk_boundary[0],
        bk_boundary[1],
        linewidth=2.0,
        color='r',
        linestyle='-.',
        coord='C',
        rot=None,
    )
    hp.projplot(
        ba_boundary[0],
        ba_boundary[1],
        linewidth=2.0,
        color='g',
        linestyle='--',
        coord='C',
        rot=None,
    )
    hp.projplot(
        spt3g_boundary[0],
        spt3g_boundary[1],
        linewidth=2.0,
        color='b',
        linestyle='-',
        coord='C',
        rot=None,
    )
    hp.graticule(True)
    title('Patch footprints')
    show()


if __name__ == '__main__':
    main()
