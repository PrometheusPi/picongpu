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

class EField
{
public:
    typedef float_X float_T;

    enum PolarizationType
    {
        /* The linear polarization of the TWTS laser is defined
         * relative to the plane of the pulse front tilt (reference plane).
         *
         * Polarisation is normal to the reference plane.
         * Use Ex-fields (and corresponding B-fields) in TWTS laser internal coordinate system.
         */
        LINEAR_X = 1u,
        /* Polarization lies within the reference plane.
         * Use Ey-fields (and corresponding B-fields) in TWTS laser internal coordinate system.
         */
        LINEAR_YZ = 2u,
    };

    /* Center of simulation volume in number of cells */
    PMACC_ALIGN(halfSimSize,DataSpace<simDim>);
    /* Laser wavelength [meter] */
    const PMACC_ALIGN(wavelength_SI,float_64);
    /* TWTS laser pulse duration [second] */
    const PMACC_ALIGN(pulselength_SI,float_64);
    /* interaction angle between TWTS laser propagation vector and the y-axis [rad] */
    const PMACC_ALIGN(phi,float_X);
    /* Takes value 1.0 for phi > 0 and -1.0 for phi < 0. */
    PMACC_ALIGN(phiPositive,float_X);
    /* propagation speed of TWTS laser overlap
    normalized to the speed of light. [Default: beta0=1.0] */
    const PMACC_ALIGN(beta_0,float_X);
    /* If auto_tdelay=FALSE, then a user defined delay is used. [second] */
    const PMACC_ALIGN(tdelay_user_SI,float_64);
    /* TWTS laser time delay */
    PMACC_ALIGN(tdelay,float_64);
    /* Should the TWTS laser delay be chosen automatically, such that
     * the laser gradually enters the simulation volume? [Default: TRUE]
     */
    const PMACC_ALIGN(auto_tdelay,bool);

    /** Electric field of the TWTS laser
     *
     * A couple of the constructor parameters are not used for the plane 
     * wave field but remain in the code in order to provide the same 
     * interface as the TWTS background field.
     * ! This may be refactored in the future !
     * 
     * Not used parameters have the name addition "_OoU" (Out of Use).
     * You can assign any number to these variables, such as 0.0.
     * 
     * \param focus_y_SI_OoU the distance to the laser focus in y-direction [m]
     * \param wavelength_SI central wavelength [m]
     * \param pulselength_SI sigma of std. gauss for intensity (E^2),
     *  pulselength_SI = FWHM_of_Intensity / 2.35482 [seconds (sigma)]
     * \param w_x_OoU beam waist: distance from the axis where the pulse electric field
     *  decreases to its 1/e^2-th part at the focus position of the laser [m]
     * \param w_y_OoU \see w_x_OoU
     * \param phi interaction angle between TWTS laser propagation vector and
     *  the y-axis [rad, default = 90.*(PI/180.)]
     * \param beta_0 propagation speed of overlap normalized to
     *  the speed of light [c, default = 1.0]
     * \param tdelay_user manual time delay if auto_tdelay is false
     * \param auto_tdelay calculate the time delay such that the TWTS pulse is not
     *  inside the simulation volume at simulation start timestep = 0 [default = true]
     * \param pol_OoU dtermines the TWTS laser polarization, which is either normal or parallel
     *  to the laser pulse front tilt plane [ default= LINEAR_X , LINEAR_YZ ]
     */
    HINLINE
    EField( const float_64 focus_y_SI_OoU,
            const float_64 wavelength_SI,
            const float_64 pulselength_SI,
            const float_64 w_x_SI_OoU,
            const float_64 w_y_SI_OoU,
            const float_X phi                = 90.*(PI / 180.),
            const float_X beta_0             = 1.0,
            const float_64 tdelay_user_SI    = 0.0,
            const bool auto_tdelay          = true,
            const PolarizationType pol_OoU = LINEAR_X );

    /** Specify your background field E(r,t) here
     *
     * \param cellIdx The total cell id counted from the start at timestep 0.
     * \param currentStep The current time step
     * \return float3_X with field normalized to amplitude in range [-1.:1.]
     */
    HDINLINE float3_X
    operator()( const DataSpace<simDim>& cellIdx,
                const uint32_t currentStep ) const;

    /** Calculate the Ex(r,t) field here (electric field vector normal to pulse-front-tilt plane)
     *
     * \param pos Spatial position of the target field
     * \param time Absolute time (SI, including all offsets and transformations)
     *  for calculating the field
     * \return Ex-field component of the non-rotated TWTS field in SI units */
    HDINLINE float_T
    calcTWTSEx( const float3_64& pos, const float_64 time ) const;



    /** Calculate the E-field vector of the TWTS laser in SI units.
     * \tparam T_dim Specializes for the simulation dimension
     * \param cellIdx The total cell id counted from the start at timestep 0
     * \return Efield vector of the rotated TWTS field in SI units */
    template <unsigned T_dim>
    HDINLINE float3_X
    getTWTSEfield_Normalized(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
            const float_64 time) const;
};

} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */
