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
#include "fields/background/templates/plane-wave-w-temporal-envelope/BField.hpp"

namespace picongpu
{
/** Load pre-defined background field */
namespace templates
{
/** Plane-wave laser with temporally increasing amplitude */
namespace pwte
{

    HINLINE
    BField::BField( const float_64 focus_y_SI_OoU,
                    const float_64 wavelength_SI,
                    const float_64 pulseduration_SI,
                    const float_64 w_x_SI_OoU,
                    const float_64 w_y_SI_OoU,
                    const float_X phi,
                    const float_X beta_0,
                    const float_64 tdelay_user_SI,
                    const bool auto_tdelay,
                    const PolarizationType pol_OoU ) :
        wavelength_SI(wavelength_SI), pulseduration_SI(pulseduration_SI), 
        phi(phi), beta_0(beta_0), tdelay_user_SI(tdelay_user_SI), 
        auto_tdelay(auto_tdelay), phiPositive( float_X(1.0) )
    {
        /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
         * on host (see fieldBackground.param), this is no problem.
         */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::getInitialTimeDelay_SI(auto_tdelay, tdelay_user_SI,
                                                halfSimSize, pulseduration_SI,
                                                focus_y_SI_OoU, phi, beta_0);
        if ( phi < float_X(0.0) ) phiPositive = float_X(-1.0);
    }

    template<>
    HDINLINE float3_X
    BField::getTWTSBfield_Normalized<DIM3>(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        typedef PMacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        PosVecVec pos(PosVecVec::create(
                                           float3_64::create(0.0)
                                       ));

        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i)
                pos[k][i] = bFieldPositions_SI[k][i];
        }

        /* An example of intra-cell position offsets is the staggered Yee-grid.
         *
         * Calculate By-component with the intra-cell offset of a By-field
         */
        const float_64 By_By = calcTWTSBy(pos[1], time);
        /* Calculate Bz-component the the intra-cell offset of a By-field */
        const float_64 Bz_By = calcTWTSBz_Ex(pos[1], time);
        /* Calculate By-component the the intra-cell offset of a Bz-field */
        const float_64 By_Bz = calcTWTSBy(pos[2], time);
        /* Calculate Bz-component the the intra-cell offset of a Bz-field */
        const float_64 Bz_Bz = calcTWTSBz_Ex(pos[2], time);
        /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex,
         * we need to back-rotate the resulting B-field vector.
         *
         * RotationMatrix[-(PI/2+phi)].(By,Bz) for rotating back the field vectors.
         */
        const float_64 By_rot = -math::sin(+phi)*By_By+math::cos(+phi)*Bz_By;
        const float_64 Bz_rot = -math::cos(+phi)*By_Bz-math::sin(+phi)*Bz_Bz;

        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X(0.0),
                         float_X(By_rot),
                         float_X(Bz_rot) );
    }



    template<>
    HDINLINE float3_X
    BField::getTWTSBfield_Normalized<DIM2>(
            const PMacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        typedef PMacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        PosVecVec pos(PosVecVec::create(
                                           float3_64::create(0.0)
                                       ));

        for (uint32_t k = 0; k<detail::numComponents;++k) {
            /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
            for (uint32_t i = 0; i<DIM2;++i)
                pos[k][i+1] = bFieldPositions_SI[k][i];
        }

        /* General background comment for the rest of this function:
         *
         * Corresponding position vector for the field components in 2D simulations.
         *  3D     3D vectors in 2D space (x, y)
         *  x -->  z (Meaning: In 2D-sim, insert cell-coordinate x
         *            into TWTS field function coordinate z.)
         *  y -->  y
         *  z --> -x (Since z=0 for 2D, we use the existing
         *            3D TWTS-field-function and set x = -0)
         *  The transformed 3D coordinates are used to calculate the field components.
         *  Ex --> Ez (Meaning: Calculate Ex-component of existing 3D TWTS-field (calcTWTSEx) using
         *             transformed position vectors to obtain the corresponding Ez-component in 2D.
         *             Note: Swapping field component coordinates also alters the
         *                   intra-cell position offset.)
         *  By --> By
         *  Bz --> -Bx (Yes, the sign is necessary.)
         *
         * An example of intra-cell position offsets is the staggered Yee-grid.
         *
         * This procedure is analogous to 3D case, but replace By --> By and Bz --> -Bx. Hence the
         * grid cell offset for Bx has to be used instead of Bz. Mind the "-"-sign.
         */

        /* Calculate By-component with the intra-cell offset of a By-field */
        const float_64 By_By =  calcTWTSBy(pos[1], time);
        /* Calculate Bx-component with the intra-cell offset of a By-field */
        const float_64 Bx_By = -calcTWTSBz_Ex(pos[1], time);
        /* Calculate By-component with the intra-cell offset of a Bx-field */
        const float_64 By_Bx =  calcTWTSBy(pos[0], time);
        /* Calculate Bx-component with the intra-cell offset of a Bx-field */
        const float_64 Bx_Bx = -calcTWTSBz_Ex(pos[0], time);
        /* Since we rotated all position vectors before calling calcTWTSBy and calcTWTSBz_Ex, we
         * need to back-rotate the resulting B-field vector. Now the rotation is done
         * analogously in the (y,x)-plane. (Reverse of the position vector transformation.)
         *
         * RotationMatrix[-(PI / 2+phi)].(By,Bx) for rotating back the field vectors.
         */
        const float_64 By_rot = -math::sin(phi)*By_By+math::cos(phi)*Bx_By;
        const float_64 Bx_rot = -math::cos(phi)*By_Bx-math::sin(phi)*Bx_Bx;

        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X(Bx_rot),
                         float_X(By_rot),
                         float_X(0.0) );
    }

 

    HDINLINE float3_X
    BField::operator()( const DataSpace<simDim>& cellIdx,
                            const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * UNIT_TIME - tdelay;
        const fieldSolver::numericalCellType::traits::FieldPosition<FieldB> fieldPosB;

        //const PMacc::math::Vector<floatD_64,detail::numComponents> bFieldPositions_SI =
        //      detail::getFieldPositions_SI(cellIdx, halfSimSize,
        //        fieldPosB(), unit_length, focus_y_SI, phi);
        
        // Remove use of focus_y compared to Alex' original implementation        
        const PMacc::math::Vector<floatD_64,detail::numComponents> bFieldPositions_SI =
              detail::getFieldPositions_SI(cellIdx, halfSimSize,
                fieldPosB(), UNIT_LENGTH, 0.0, phi);
        
        return getTWTSBfield_Normalized<simDim>(bFieldPositions_SI, time_SI);
    }



    /** Calculate the By(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcTWTSBy( const float3_64& pos, const float_64 time ) const
    {
        /* Normalize width of temporal envelope.
         * factor 2  in tauG arises from definition convention in laser formula 
         * */
        const float_T tauG = float_T(pulseduration_SI*2.0 / UNIT_TIME);

        /*
         * Implementation of the plane wave formula for the magnetic field.
         * If you change this, you have to change the implementation for
         * the electric field as well!
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

        return temp_envelope*sin(phase) / UNIT_SPEED;
    }



    /** Calculate the Bz(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcTWTSBz_Ex( const float3_64& pos, const float_64 time ) const
    {
        return float_T(0.0) / UNIT_SPEED;
    }

} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */

