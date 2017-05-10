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
    EField::EField( const float_64 focus_y_SI,
                    const float_64 wavelength_SI,
                    const float_64 pulselength_SI,
                    const float_64 w_x_SI,
                    const float_64 w_y_SI,
                    const float_X phi,
                    const float_X beta_0,
                    const float_64 tdelay_user_SI,
                    const bool auto_tdelay,
                    const PolarizationType pol ) :
        focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
        pulselength_SI(pulselength_SI), w_x_SI(w_x_SI),
        w_y_SI(w_y_SI), phi(phi), beta_0(beta_0),
        tdelay_user_SI(tdelay_user_SI), dt(SI::DELTA_T_SI),
        unit_length(UNIT_LENGTH), auto_tdelay(auto_tdelay), pol(pol), phiPositive( float_X(1.0) )
    {
        /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                 on host (see fieldBackground.param), this is no problem.
         */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::getInitialTimeDelay_SI(auto_tdelay, tdelay_user_SI,
                                                halfSimSize, pulselength_SI,
                                                focus_y_SI, phi, beta_0);
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

    //template<>
    //HDINLINE float3_X
    //EField::getTWTSEfield_Normalized_Ey<DIM3>(
                //const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
                //const float_64 time) const
    //{
        //typedef PMacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        //PosVecVec pos(PosVecVec::create(
                                           //float3_64::create(0.0)
                                       //));

        //for (uint32_t k = 0; k<detail::numComponents;++k) {
            //for (uint32_t i = 0; i<simDim;++i) pos[k][i] = eFieldPositions_SI[k][i];
        //}

        ///* Calculate Ey-component with the intra-cell offset of a Ey-field */
        //const float_64 Ey_Ey = calcTWTSEy(pos[1], time);
        ///* Calculate Ey-component with the intra-cell offset of a Ez-field */
        //const float_64 Ey_Ez = calcTWTSEy(pos[2], time);

        ///* Since we rotated all position vectors before calling calcTWTSEy,
         //* we need to back-rotate the resulting E-field vector.
         //*
         //* RotationMatrix[-(PI/2+phi)].(Ey,Ez) for rotating back the field-vectors.
         //*/
        //const float_64 Ey_rot = -math::sin(+phi)*Ey_Ey;
        //const float_64 Ez_rot = -math::cos(+phi)*Ey_Ez;

        ///* Finally, the E-field normalized to the peak amplitude. */
        //return float3_X( float_X(0.0),
                         //float_X(Ey_rot),
                         //float_X(Ez_rot) );
    //}

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

    //template<>
    //HDINLINE float3_X
    //EField::getTWTSEfield_Normalized_Ey<DIM2>(
        //const PMacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
        //const float_64 time) const
    //{
        //typedef PMacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        //PosVecVec pos(PosVecVec::create(
                                           //float3_64::create(0.0)
                                       //));

        ///* The 2D output of getFieldPositions_SI only returns
         //* the y- and z-component of a 3D vector.
         //*/
        //for (uint32_t k = 0; k<detail::numComponents;++k) {
            //for (uint32_t i = 0; i<DIM2;++i) pos[k][i+1] = eFieldPositions_SI[k][i];
        //}

        ///* Ey->Ey, but grid cell offsets for Ex and Ey have to be used.
         //*
         //* Calculate Ey-component with the intra-cell offset of a Ey-field
         //*/
        //const float_64 Ey_Ey = calcTWTSEy(pos[1], time);
        ///* Calculate Ey-component with the intra-cell offset of a Ex-field */
        //const float_64 Ey_Ex = calcTWTSEy(pos[0], time);

        ///* Since we rotated all position vectors before calling calcTWTSEy,
         //* we need to back-rotate the resulting E-field vector.
         //*
         //* RotationMatrix[-(PI / 2+phi)].(Ey,Ex) for rotating back the field-vectors.
         //*/
        //const float_64 Ey_rot = -math::sin(+phi)*Ey_Ey;
        //const float_64 Ex_rot = -math::cos(+phi)*Ey_Ex;

        ///* Finally, the E-field normalized to the peak amplitude. */
        //return float3_X( float_X(Ex_rot),
                         //float_X(Ey_rot),
                         //float_X(0.0) );
    //}

    HDINLINE float3_X
    EField::operator()( const DataSpace<simDim>& cellIdx,
                            const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * dt - tdelay;
        const fieldSolver::numericalCellType::traits::FieldPosition<FieldE> fieldPosE;

        const PMacc::math::Vector<floatD_64,detail::numComponents> eFieldPositions_SI =
              detail::getFieldPositions_SI(cellIdx, halfSimSize,
                fieldPosE(), unit_length, focus_y_SI, phi);

        /* Single TWTS-Pulse */
        //switch (pol)
        //{
            //case LINEAR_X :
            //return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI);

            //case LINEAR_YZ :
            //return getTWTSEfield_Normalized_Ey<simDim>(eFieldPositions_SI, time_SI);
        //}
        return getTWTSEfield_Normalized<simDim>(eFieldPositions_SI, time_SI); // defensive default
    }

    /** Calculate the Ex(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcTWTSEx( const float3_64& pos, const float_64 time) const
    {
        typedef PMacc::math::Complex<float_T> complex_T;
        typedef PMacc::math::Complex<float_64> complex_64;
        /** Unit of Speed */
        const double UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /** Unit of time */
        const double UNIT_TIME = SI::DELTA_T_SI;
        /** Unit of length */
        const double UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;
    
        /* If phi < 0 the formulas below are not directly applicable.
         * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
         * z-axis of the coordinate system in this function.
         */
        const float_T phiReal = float_T( math::abs(phi) );
        const float_T alphaTilt = math::atan2(float_T(1.0)-float_T(beta_0)*math::cos(phiReal),
                                                float_T(beta_0)*math::sin(phiReal));
        /* Definition of the laser pulse front tilt angle for the laser field below.
         *
         * For beta0 = 1.0, this is equivalent to our standard definition. Question: Why is the
         * local "phi_T" not equal in value to the object member "phiReal" or "phi"?
         * Because the standard TWTS pulse is defined for beta0 = 1.0 and in the coordinate-system
         * of the TWTS model phi is responsible for pulse front tilt and dispersion only. Hence
         * the dispersion will (although physically correct) be slightly off the ideal TWTS
         * pulse for beta0 != 1.0. This only shows that this TWTS pulse is primarily designed for
         * scenarios close to beta0 = 1.
         */
        const float_T phiT = float_T(2.0)*alphaTilt;

        /* Angle between the laser pulse front and the y-axis. Not used, but remains in code for
         * documentation purposes.
         * const float_T eta = (PI / 2) - (phiReal - alphaTilt);
         */

        const float_T cspeed = float_T( SI::SPEED_OF_LIGHT_SI / UNIT_SPEED );
        const float_T lambda0 = float_T(wavelength_SI / UNIT_LENGTH);
        const float_T om0 = float_T(2.0*PI*cspeed / lambda0);
        /* factor 2  in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T(pulselength_SI*2.0 / UNIT_TIME);
        /* w0 is wx here --> w0 could be replaced by wx */
        const float_T w0 = float_T(w_x_SI / UNIT_LENGTH);
        const float_T rho0 = float_T(PI*w0*w0/lambda0);
        /* wy is width of TWTS pulse */
        const float_T wy = float_T(w_y_SI / UNIT_LENGTH);
        const float_T k = float_T(2.0*PI / lambda0);

        /* In order to calculate in single-precision and in order to account for errors in
         * the approximations far from the coordinate origin, we use the wavelength-periodicity and
         * the known propagation direction for realizing the laser pulse using relative coordinates
         * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
         * in double precision.
         */
        const float_64 tanAlpha = ( float_64(1.0) - beta_0 * math::cos(phi) )
                                    / ( beta_0 * math::sin(phi) );
        const float_64 tanFocalLine = math::tan( PI / float_64(2.0) - phi );
        const float_64 deltaT = wavelength_SI / SI::SPEED_OF_LIGHT_SI
                                 * ( float_64(1.0) + tanAlpha / tanFocalLine);
        const float_64 deltaY = wavelength_SI / tanFocalLine;
        const float_64 deltaZ = -wavelength_SI;
        const float_64 numberOfPeriods = math::floor( time / deltaT );
        const float_T timeMod = float_T( time - numberOfPeriods * deltaT );
        const float_T yMod = float_T( pos.y() + numberOfPeriods * deltaY );
        const float_T zMod = float_T( pos.z() + numberOfPeriods * deltaZ );

        const float_T x = float_T(phiPositive * pos.x() / UNIT_LENGTH);
        const float_T y = float_T(phiPositive * yMod / UNIT_LENGTH);
        const float_T z = float_T(zMod / UNIT_LENGTH);
        const float_T t = float_T(timeMod / UNIT_TIME);

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
        const float_T temp_slope = float_T(2./(.5*tauG)); // use for tanh-startup
        
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
        //const float_T temp_envelope = float_T(1.0)
        //    /(float_T(1.0) + math::exp(-envelope_phase)); // tanh-startup
        const float_T temp_envelope = float_T(1.0);

        // Phase of the plane wave electric field travelling along +z
        const float_T phase = k*(cspeed*t - z);

        return temp_envelope*sin(phase);
    }

    ///** Calculate the Ey(r,t) field here
     //*
     //* \param pos Spatial position of the target field.
     //* \param time Absolute time (SI, including all offsets and transformations) for calculating
     //*             the field */
    //HDINLINE EField::float_T
    //EField::calcTWTSEy( const float3_64& pos, const float_64 time) const
    //{
        ///* The field function of Ey (polarization in pulse-front-tilt plane)
         //* is by definition identical to Ex (polarization normal to pulse-front-tilt plane)
         //*/
        //return calcTWTSEx( pos, time );
    //}

} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */

