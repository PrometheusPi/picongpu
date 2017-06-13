/**
 * Copyright 2014-2017 Alexander Debus, Axel Huebl, Klaus Steiniger
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * This is a 'quick & dirty' implementation of a plane wave background 
 * field with temporally increasing amplitude.
 * 'Quick & dirty' means that much of the code is copy & pasted from the
 * TWTS background field implementation.
 * Along the lines of 'Whatever you do twice: Automate!' both 
 * implementations need to be refactored to outsource commonly used 
 * functionalitites.
 */

#pragma once

#include "pmacc_types.hpp"

#include "math/Vector.hpp"
#include "dimensions/DataSpace.hpp"
#include "fields/background/templates/plane-wave-w-temporal-envelope/numComponents.hpp"

namespace picongpu
{
/* Load pre-defined background field */
namespace templates
{
/* Plane-wave laser with temporally increasing amplitude */
namespace pwte
{

class BField
{
public:
    typedef float_X float_T;

    enum PolarizationType
    {
        /* The linear polarization of the plane wave laser is defined
         * relative to the plane spanned by electron and laser 
         * propagation directions (reference plane).
         *
         * Polarisation in Ex is normal to the reference plane.
         */
        LINEAR_X = 1u,
        /* Polarization in Ey and/or Ez lies within the reference plane.
         */
        LINEAR_YZ = 2u,
    };

    /* Center of simulation volume in number of cells */
    PMACC_ALIGN(halfSimSize,DataSpace<simDim>);
    /* Laser wavelength [meter] */
    const PMACC_ALIGN(wavelength_SI,float_64);
    /* Duration of a gaussian laser pulse with approximately 
     * equal temporal slope.
     * Used to define the slope the of the temporal envelope.
     * [second] */
    const PMACC_ALIGN(pulseduration_SI,float_64);
    /* interaction angle enclosed by plane wave laser propagation 
     * vector and the y-axis (electron propagation direction)
     * [rad] */
    const PMACC_ALIGN(phi,float_X);
    /* Takes value 1.0 for phi > 0 and -1.0 for phi < 0. */
    PMACC_ALIGN(phiPositive,float_X);
    /* If auto_tdelay=FALSE, then a user defined delay is used. [second] */
    const PMACC_ALIGN(tdelay_user_SI,float_64);
    /* Temporal envelope time delay.
     * Without time delay the laser amplitude is already at half of its 
     * maximum value at simulation start.*/
    PMACC_ALIGN(tdelay,float_64);
    /* Should the plane wave laser time delay be chosen automatically, 
     * such that the laser amplitude starts at zero at the beginning 
     * of the simulation and then gradually increases? [Default: TRUE]
     */
    const PMACC_ALIGN(auto_tdelay,bool);



    /** Magnetic field of the plane wave laser
     *
     * A couple of the constructor parameters are not used for the plane 
     * wave field but remain in the code in order to provide the same 
     * interface as the TWTS background field.
     * **This may be refactored in the future**
     * 
     * Not used parameters have the name addition "_OoU" (Out of Use).
     * You can assign any number to these variables, such as 0.0.
     * 
     * \param focus_y_SI_OoU the distance to the laser focus in y-direction [m]
     * \param wavelength_SI central wavelength [m]
     * \param pulseduration_SI Defines the the temporal slope of the 
     *  temporal envelope.
     *  The temporal envelope is modelled by a tanh function, i.e.
     *    1 / (1 + exp[ -4*(t - t0)/tauG ]),
     *  where the slope 4/tauG is choosen to resemble the slope of 
     *  the field of a gaussian pulse of the form exp[-(t/tauG)^2].
     *  pulseduration_SI is the sigma of the intensity (E^2) of this gaussian 
     *  field,
     *  pulseduration_SI = FWHM_of_Intensity / 2.35482 [seconds (sigma)].
     *  With this,
     *    tauG = FWHM_of_Intensity / sqrt[2 * ln(2)].
     * \param w_x_OoU beam waist: distance from the axis where the pulse electric field
     *  decreases to its 1/e^2-th part at the focus position of the laser [m]
     * \param w_y_OoU \see w_x_OoU
     * \param phi interaction angle between plane wave laser propagation vector and
     *  the y-axis [rad, default = 90.*(PI/180.)]. 
     *  These vectors span the interaction plane.
     * \param beta_0_OoU propagation speed of laser pulse and electron bunch
     *  overlap region. 
     *  Normalized to the speed of light [c, default = 1.0]
     * \param tdelay_user_SI manual time delay if auto_tdelay is false
     *  [seconds]
     * \param auto_tdelay calculate the time delay such that the 
     *  plane wave laser starts at (approximately) zero amplitude 
     *  at simulation start timestep = 0 [default = true]
     * \param pol_OoU dtermines the plane wave laser polarization, 
     *  which is either normal or parallel
     *  to the interaction plane [ default= LINEAR_X , LINEAR_YZ ]
     */
    HINLINE
    BField( const float_64 focus_y_SI_OoU,
            const float_64 wavelength_SI,
            const float_64 pulseduration_SI,
            const float_64 w_x_SI_OoU,
            const float_64 w_y_SI_OoU,
            const float_X phi               = 90.*(PI / 180.),
            const float_X beta_0_OoU        = 1.0,
            const float_64 tdelay_user_SI   = 0.0,
            const bool auto_tdelay          = true,
            const PolarizationType pol_OoU = LINEAR_X );



    /** Specify your background field B(r,t) here
     *
     * \param cellIdx The total cell id counted from the start at t=0
     * \param currentStep The current time step */
    HDINLINE float3_X
    operator()( const DataSpace<simDim>& cellIdx,
                const uint32_t currentStep ) const;



    /** Calculate the By(r,t) field, when electric field vector (Ex,0,0)
     *  is normal to the pulse-front-tilt plane (y,z)
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field */
    HDINLINE float_T
    calcTWTSBy( const float3_64& pos, const float_64 time ) const;



    /** Calculate the Bz(r,t) field, when electric field vector (Ex,0,0)
     *  is normal to the pulse-front-tilt plane (y,z)
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field */
    HDINLINE float_T
    calcTWTSBz_Ex( const float3_64& pos, const float_64 time ) const;



    /** Calculate the B-field vector of the plane wave laser in SI units.
     * \tparam T_dim Specializes for the simulation dimension
     * \param cellIdx The total cell id counted from the start at timestep 0
     * \return B-field vector of the rotated plane wave field in SI units */
    template<unsigned T_dim>
    HDINLINE float3_X
    getTWTSBfield_Normalized(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
            const float_64 time) const;
};

} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */
