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

namespace picongpu
{
namespace templates
{
namespace pwte
{
/* Auxiliary functions for calculating the plane wave laser */
namespace detail
{

    template <unsigned T_dim>
    class GetInitialTimeDelay
    {
        public:
        /** Obtain the SI time delay that later enters the Ex(r, t), By(r, t) and Bz(r, t)
         *  calculations as t.
         * 
         * A couple of the constructor parameters are not used for the plane 
         * wave field but remain in the code in order to provide the same 
         * interface as the TWTS background field.
         * ! This may be refactored in the future !
         * 
         * Not used parameters have the name addition "_OoU" (Out of Use).
         * You can assign any number to these variables, such as 0.0.
         * 
         * \tparam T_dim Specializes for the simulation dimension
         *  \param auto_tdelay calculate the time delay such that the 
         *      plane wave laser is close at simulation start timestep = 0 [default = true]
         *  \param tdelay_user_SI manual time delay if auto_tdelay is false
         *  \param halfSimSize center of simulation volume in number of cells
         *  \param pulseduration_SI Defines the the temporal slope of the 
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
         *  \param focus_y_SI_OoU the distance to the laser focus in y-direction [m]
         *  \param phi interaction angle enclosed by laser propagation vector and
         *      the y-axis (= electron propagation direction) 
         *      unit: [rad]
         *  \param beta_0 propagation speed of overlap normalized
         *      to the speed of light [c, default = 1.0]
         *  \return time delay in SI units */
        HDINLINE float_64 operator()( const bool auto_tdelay,
                                      const float_64 tdelay_user_SI,
                                      const DataSpace<simDim>& halfSimSize,
                                      const float_64 pulseduration_SI,
                                      const float_64 focus_y_SI_OoU,
                                      const float_X phi,
                                      const float_X beta_0 ) const;
    };

    template<>
    HDINLINE float_64
    GetInitialTimeDelay<DIM3>::operator()( const bool auto_tdelay,
                                           const float_64 tdelay_user_SI,
                                           const DataSpace<simDim>& halfSimSize,
                                           const float_64 pulseduration_SI,
                                           const float_64 focus_y_SI_OoU,
                                           const float_X phi,
                                           const float_X beta_0 ) const
    {
        if ( auto_tdelay ) {

            /* Fudge parameter to make sure, that temporal envelope 
             * (and thus field amplitude) is
             * close to zero at simulation begin. */
            const float_64 m = 3.;
            /* Laser pulse duration coreesponding to Field \propto exp^{-t^2/tauG^2}*/
            const float_64 tauG = 2.0 * pulseduration_SI;
            /* Programmatically obtained time-delay */
            const float_64 tdelay = m*tauG;

            return tdelay;
        }
        else
            return tdelay_user_SI;
    }

    template <>
    HDINLINE float_64
    GetInitialTimeDelay<DIM2>::operator()( const bool auto_tdelay,
                                           const float_64 tdelay_user_SI,
                                           const DataSpace<simDim>& halfSimSize,
                                           const float_64 pulseduration_SI,
                                           const float_64 focus_y_SI_OoU,
                                           const float_X phi,
                                           const float_X beta_0 ) const
    {
        if ( auto_tdelay ) {

            /* Fudge parameter to make sure, that temporal envelope 
             * (and thus field amplitude) is
             * close to zero at simulation begin. */
            const float_64 m = 3.;
            /* Laser pulse duration coreesponding to Field \propto exp^{-t^2/tauG^2}*/
            const float_64 tauG = 2.0 * pulseduration_SI;
            /* Programmatically obtained time-delay */
            const float_64 tdelay = m*tauG;

            return tdelay;
        }
        else
            return tdelay_user_SI;
    }

    template <unsigned T_Dim>
    HDINLINE float_64
    getInitialTimeDelay_SI( const bool auto_tdelay,
                            const float_64 tdelay_user_SI,
                            const DataSpace<T_Dim>& halfSimSize,
                            const float_64 pulseduration_SI,
                            const float_64 focus_y_SI_OoU,
                            const float_X phi,
                            const float_X beta_0 )
    {
        return GetInitialTimeDelay<T_Dim>()(auto_tdelay, tdelay_user_SI,
                                            halfSimSize, pulseduration_SI,
                                            focus_y_SI_OoU, phi, beta_0);
    }

} /* namespace detail */
} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */
