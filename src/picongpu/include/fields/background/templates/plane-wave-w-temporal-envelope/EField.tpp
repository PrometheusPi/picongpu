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
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"

#include "math/Vector.hpp"
#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "math/Complex.hpp"

#include "fields/background/templates/plane-wave-w-temporal-envelope/RotateField.tpp"
#include "fields/background/templates/plane-wave-w-temporal-envelope/GetInitialTimeDelay_SI.tpp"
#include "fields/background/templates/plane-wave-w-temporal-envelope/getFieldPositions_SI.tpp"
#include "fields/background/templates/plane-wave-w-temporal-envelope/EField.hpp"

namespace picongpu
{
/* Load pre-defined background field */
namespace templates
{
/* Plane-wave laser with temporally increasing amplitude */
namespace pwte
{

    HINLINE
    EField::EField( const float_64 focus_y_SI_OoU,
                    const float_64 wavelength_SI,
                    const float_64 pulselength_SI,
                    const float_64 w_x_SI_OoU,
                    const float_64 w_y_SI_OoU,
                    const float_X phi,
                    const float_X beta_0,
                    const float_64 tdelay_user_SI,
                    const bool auto_tdelay,
                    const PolarizationType pol_OoU ) :
        wavelength_SI(wavelength_SI), pulselength_SI(pulselength_SI), 
        phi(phi), beta_0(beta_0), tdelay_user_SI(tdelay_user_SI),
        auto_tdelay(auto_tdelay), phiPositive( float_X(1.0) )
    {
        /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                 on host (see fieldBackground.param), this is no problem.
         */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::getInitialTimeDelay_SI(auto_tdelay, tdelay_user_SI,
                                                halfSimSize, pulselength_SI,
                                                focus_y_SI_OoU, phi, beta_0);
        if ( phi < float_X(0.0) ) phiPositive = float_X(-1.0);
    }



    template<>
    HDINLINE float3_X
    EField::getTWTSEfield_Normalized<DIM3>(
                const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
    {
        float3_64 pos(float3_64::create(0.0));
        for (uint32_t i = 0; i<simDim;++i) pos[i] = eFieldPositions_SI[0][i];
        return float3_X( float_X( calcTWTSEx(pos,time) ),
                         float_X(0.), float_X(0.) );
    }



    template<>
    HDINLINE float3_X
    EField::getTWTSEfield_Normalized<DIM2>(
        const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
        const float_64 time) const
    {
        /* Ex->Ez, so also the grid cell offset for Ez has to be used. */
        float3_64 pos(float3_64::create(0.0));
        /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
        for (uint32_t i = 0; i<DIM2;++i) pos[i+1] = eFieldPositions_SI[2][i];
        return float3_X( float_X(0.), float_X(0.),
                         float_X( calcTWTSEx(pos,time) ) );
    }



    HDINLINE float3_X
    EField::operator()( const DataSpace<simDim>& cellIdx,
                            const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * UNIT_TIME - tdelay;
        const fieldSolver::numericalCellType::traits::FieldPosition<FieldE> fieldPosE;

        // Vector r is 3-dim of float_64 numbers
        //const PMacc::math::Vector<floatD_64,detail::numComponents> eFieldPositions_SI =
        //      detail::getFieldPositions_SI(cellIdx, halfSimSize,
        //        fieldPosE(), UNIT_LENGTH, focus_y_SI, phi);

        // Remove use of focus_y compared to Alex' original implementation
        const PMacc::math::Vector<floatD_64,detail::numComponents> eFieldPositions_SI =
              detail::getFieldPositions_SI(cellIdx, halfSimSize,
                fieldPosE(), UNIT_LENGTH, 0.0, phi);

        return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);
    }



    /** Calculate the Ex(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcTWTSEx( const float3_64& pos, const float_64 time) const
    {
        /* Normalize width of temporal envelope.
         * Factor 2 in tauG arises from definition convention in laser formula 
         * */
        const float_T tauG = float_T(pulselength_SI*2.0 / UNIT_TIME);
 
        /*
         * Implementation of the plane wave formula for the electric field.
         * If you change this, you have to change the implementation for
         * the magnetic field as well!
         */ 

        // Choosen such that the smooth step like envelope reaches its 
        // maximum at approximately t=0 similar to the behaviour of a 
        // (standard) gaussian envelope.
        // This exact value is defined by requesting the gauss and the 
        // step envelope to reach a value of 0.5 at the same time. 
        const float_T temp_offset = - math::sqrt(math::log(float_T(2.0)))*tauG;

        // Slope of the temporal envelope
        // The value is choosen at whim with the aim to resemble the 
        // slope of a gaussian envelope
        //const float_T temp_slope = float_T(1./(.7*tauG)); // use for erf-startup
        const float_T temp_slope = float_T(4./tauG); // use for tanh-startup
        
        // Determine from the current simulation time the proper phase 
        // of the envelopes temporal evolution
        const float_T envelope_phase = 
            temp_slope*(float_T(time/UNIT_TIME) - temp_offset);
        
        // Smooth step-like temporal envelope by an error function or
        // a tangens hyperbolicus.
        // Both reache the value 0.5 at the same time a gaussian envelope of
        // the form exp(-t**2/tauG**2) reaches 0.5. 
        //const float_T temp_envelope = float_T(0.5)
        //    *(float_T(1.0) + math::erf(temp_slope*(t - temp_offset))); // erf-startup
        const float_T temp_envelope = float_T(1.0)
            /(float_T(1.0) + math::exp(-envelope_phase)); // tanh-startup

        // Phase of the plane wave electric field travelling along +z
        const float_T phase = 
            float_T((2.*PI)*((SI::SPEED_OF_LIGHT_SI*time - pos.z())/wavelength_SI));

        return temp_envelope*sin(phase);
    }

} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */

