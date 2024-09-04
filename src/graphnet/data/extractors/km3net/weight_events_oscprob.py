#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ROOT
import numpy as np
import km3flux
from .utils_osc import _zipequalize, _pdgid2flavor
import argparse


OscProbDir = "/pbs/throng/km3net/software/oscprob/1.3"
osc_prob_lib = OscProbDir + "/libOscProb.so"
ROOT.gSystem.Load(osc_prob_lib)
op = ROOT.OscProb

def _osc_prob(pmns, prem, flav_in, flav_out, energies, cos_zenith):
    """Execution of OscProb

    Args:
        pmns (pmns object): PMNS object from OscProb containing the oscillation parameters
        prem (prem object): PREM object from OscProb containing the earth model
        flav_in (array): Initial neutrino flavor
        flav_out (array): Final neutrino flavor
        energies (array): Neutrino energies
        cos_zenith (array): Cosine of the zenith angle

    Returns:
        array: The oscillation probabilites for each event.
    """
    params = _zipequalize(flav_in, flav_out, energies, cos_zenith)

    # use a PremModel to make the paths through the earth
    # with the class PremModel
    # chose an angle for the neutrino and fill the paths with cosTheta,
    # e.g. cosTheta = -1 (vertical up-going)
    P = []
    for fl_in, fl_out, E, cosZ in zip(*params):
        if fl_in < 0:
            pmns.SetIsNuBar(True)
        else:
            pmns.SetIsNuBar(False)
        prem.FillPath(cosZ)
        pmns.SetPath(prem.GetNuPath())
        P.append(pmns.Prob(_pdgid2flavor(fl_in), _pdgid2flavor(fl_out), E))
    return P


def use_oscprob(flavour_in, flavour_out, energies, cos_zeniths, nu_params=None):
    """
    Interface to the oscprob library. Initialization of oscillation parameters and calculation of oscillation probabilities.
    
    Parameters
    ----------
    flavour_in : array
        Initial neutrino flavour
    flavour_out : array
        Final neutrino flavour
    energies : array
        Neutrino energies
    cos_zeniths : array
        Cosine of the zenith angle
    nu_params : dict
        A dict to give specific oscillation parameters. Default are the current best nufit values.

    Returns
    -------
    p : array
        The oscillation probabilities for each event.
    """

    # oscprob config
    pmns = op.PMNS_Fast()
    prem = op.PremModel()

    ORCA_DEPTH_KM = 2.4  # km
    prem.SetDetPos(6371.0 - ORCA_DEPTH_KM)  # values from Zineb

    # nufit parameters with SK atmospheric data. values from Nufit2022 in http://www.nu-fit.org/?q=node/256
    if nu_params is None:
        nu_params = {
            "dm_21": 7.41e-5,
            "dm_31": 2.507e-3,  # if not inv_hierarchy else dm_21 - 2.465e-3
            "theta_12": 33.41 * np.pi / 180,
            "theta_23": 42.2 * np.pi / 180,  # if inverted ordering 49.0 * np.pi/180
            "theta_13": 8.58 * np.pi / 180,  # if inverted ordering 8.57 * np.pi/180
            "dcp": 232 * np.pi / 180,  # if inverted ordering 276 * np.pi / 180
        }

    pmns.SetDm(2, nu_params["dm_21"])  # set delta_m21 in eV^2
    pmns.SetDm(3, nu_params["dm_31"])  # set delta_m31 in eV^2
    pmns.SetAngle(1, 2, nu_params["theta_12"])  # set Theta12 in radians
    pmns.SetAngle(1, 3, nu_params["theta_13"])  # set Theta13 in radians
    pmns.SetAngle(2, 3, nu_params["theta_23"])  # set Theta23 in radians
    pmns.SetDelta(1, 3, nu_params["dcp"])  # set Delta13 in radians

    p = _osc_prob(
        pmns,
        prem,
        flavour_in,
        flavour_out,
        energies,
        cos_zeniths,
    )

    return p

def compute_osc_weight(nu_type, energy, dir_z, is_cc, oscillation=True, nu_params=None):
    """
    Computes oscillation weight based on event properties

    Parameters
    ----------
    nu_type : array
        The flavors (pdgid) of the particles for which to get the weights.
    energy : array
        The energies of the interactions.
    dir_z : array
        The z-dir, cos(theta), of the interaction.
    is_cc : array
        The description of the current of the interaction: 2 is cc, 3 is nc.
    oscillation : bool
        True for consider oscillations, false for not
    nu_params : dict
        A dict to give specific oscillation parameters. Default are the current best nufit values.

    Returns
    -------
    weight : array
        The oscillation weights for each event.
    """
    honda = km3flux.flux.Honda()

    # make sure these are all arrays (there were problems with data frames)
    nu_type = np.array(nu_type)
    energy = np.array(energy)
    dir_z = np.array(dir_z)
    is_cc = np.array(is_cc)

    nu_dict = {
        12: honda.flux(2014, "Frejus", solar="min", averaged="azimuth")["nue"],
        14: honda.flux(2014, "Frejus", solar="min", averaged="azimuth")["numu"],
        -12: honda.flux(2014, "Frejus", solar="min", averaged="azimuth")["anue"],
        -14: honda.flux(2014, "Frejus", solar="min", averaged="azimuth")["anumu"],
    }

    # nufit parameters with SK atmospheric data. values from Nufit2022 in http://www.nu-fit.org/?q=node/256
    if nu_params is None:
        nu_params = {
            "dm_21": 7.42e-5,
            "dm_31": 2.510e-3,
            "theta_12": 33.45 * np.pi / 180,
            "theta_23": 42.1 * np.pi / 180,
            "theta_13": 8.62 * np.pi / 180,
            "dcp": 230 * np.pi / 180,
        }

    weight = np.zeros(len(nu_type))

    # make sure the correct is_cc convention is used
    cc_mask = (
        is_cc == 1
    )  # careful definition of is_cc
    nc_mask = np.invert(cc_mask)

    for flav in [12, 14]:

        # get oscillation probs for CC events
        if oscillation:
            # convention: use - dir_z for this
            # flavour_in,flavour_out,...

            osc_prob_cc = use_oscprob(
                flav * np.sign(nu_type[cc_mask]),
                nu_type[cc_mask],
                energy[cc_mask],
                -dir_z[cc_mask],
                nu_params,
            )

        else:
            osc_prob_cc = np.zeros(np.count_nonzero(cc_mask))
            flavor_out_is_same_as_flavor_in_mask = (
                flav * np.sign(nu_type[cc_mask]) == nu_type[cc_mask]
            )
            osc_prob_cc[flavor_out_is_same_as_flavor_in_mask] = np.ones(
                np.count_nonzero(flavor_out_is_same_as_flavor_in_mask)
            )

        # get flux in a loop as it only takes single numbers (as of yet)
        flux = []
        names = np.vectorize(nu_dict.get)(flav * np.sign(nu_type))
        flux = [
            nu_dict[flav * np.sign(nu_type[i])](energy[i], dir_z[i])
            for i in range(len(nu_type))
        ]

        flux = np.asarray(flux)

        # set all the CC weights: osc_prob times flux
        weight[cc_mask] += osc_prob_cc * flux[cc_mask]


        # set all NC weights; total flux, consists of e & mu, no oscillations
        weight[nc_mask] += flux[nc_mask]

    return weight


def compute_evt_weight(
    nu_type, energy, dir_z, is_cc, w2, livetime, ngen, oscillation=True, nu_params=None
):
    """
    Computes event weight with oscillations

    Parameters
    ----------
    nu_type : array
        The flavors (pdgid) of the particles for which to get the weights.
    energy : array
        The energies of the interactions.
    dir_z : array
        The z-dir, cos(theta), of the interaction.
    is_cc : array
        The description of the current of the interaction: 2 is cc, 3 is nc.
    oscillation : bool
        True for consider oscillations, false for not
    nu_params : dict
        A dict to give specific oscillation parameters. Default are the current best nufit values.

    Returns
    -------
    weight : array
        The weights for each event including the oscillations.
    """

    # make sure these are all arrays (there were problems with data frames)
    nu_type = np.array(nu_type)
    energy = np.array(energy)
    dir_z = np.array(dir_z)
    is_cc = np.array(is_cc)
    w2 = np.array(w2)
    livetime = np.array(livetime)
    ngen = np.array(ngen)

    oscilation_w = compute_osc_weight(
        nu_type, energy, dir_z, is_cc, oscillation, nu_params
    )
    weight = livetime * w2 / ngen * oscilation_w

    return weight

