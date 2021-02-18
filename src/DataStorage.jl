module DataStorage

## Imports
import Base: size #must import to add a definition to size

##Exports
export DataContainer, PairedDataContainer
export size
export get_data, get_inputs, get_outputs

## Objects
"""
    struct DataContainer

struct to store data samples as columns in an array
"""
struct DataContainer{FT <: Real}
    #stored data, each piece of data is a column [number of samples x data size]
    stored_data::Array{FT,2}    
    #constructor with 2D arrays
    function DataContainer(
        stored_data::Array{FT,2};
        data_are_columns=true) where {FT <: Real}

        if data_are_columns
            new{FT}(stored_data)
        else
            new{FT}(permutedims(stored_data,(2,1)))
        end
    end
end

"""
    PairedDataContainer

stores input - output pairs as data containers
"""
struct PairedDataContainer{FT <: Real}
    #container for inputs and ouputs [sample_size x data/parameter size]
    inputs::DataContainer{FT} 
    outputs::DataContainer{FT}

    #constructor with 2D Arrays
    function PairedDataContainer(
        inputs::Array{FT,2},
        outputs::Array{FT,2};
        data_are_columns=true) where {FT <: Real}

        sample_dim = data_are_columns ? 2 : 1
        if !(size(inputs,sample_dim) == size(outputs,sample_dim))
                throw(DimensionMismatch("There must be the same number of samples of both inputs and outputs"))
        end
        
        stored_inputs = DataContainer(inputs; data_are_columns=data_are_columns)
        stored_outputs = DataContainer(outputs; data_are_columns=data_are_columns)
        new{FT}(stored_inputs,stored_outputs)
        
    end
    #constructor with DataContainers
    function PairedDataContainer(
        inputs::DataContainer,
        outputs::DataContainer)

        if !(size(inputs,2) == size(outputs,2))
            throw(DimensionMismatch("There must be the same number of samples of both inputs and outputs"))    
        else
            FT = eltype(get_data(inputs))
            new{FT}(inputs,outputs)
        end
    end
    
end

#other constructors


## functions
function size(dc::DataContainer)
    return size(dc.stored_data)
end

function size(dc::DataContainer,idx::IT) where {IT <: Integer}
    return size(dc.stored_data,idx)
end

function size(pdc::PairedDataContainer)
    return size(pdc.inputs), size(pdc.outputs)
end

function size(pdc::PairedDataContainer,idx::IT) where {IT <: Integer}
    return size(pdc.inputs,idx), size(pdc.outputs,idx)
end

function get_data(dc::DataContainer)
    return dc.stored_data
end
function get_data(pdc::PairedDataContainer)
    return get_inputs(pdc), get_outputs(pdc)
end

function get_inputs(pdc::PairedDataContainer)
    return get_data(pdc.inputs)
end
function get_outputs(pdc::PairedDataContainer)
    return get_data(pdc.outputs)
end


end
