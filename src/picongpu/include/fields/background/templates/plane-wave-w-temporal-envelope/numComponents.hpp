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

namespace picongpu
{
namespace templates
{
namespace pwte
{
namespace detail
{
    /* Number of field components used in the simulation. [Default: 3 for both 2D and 3D] */
    const uint32_t numComponents = 3;
} /* namespace detail */
} /* namespace pwte */
} /* namespace templates */
} /* namespace picongpu */
