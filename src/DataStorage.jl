module DataStorage

## Imports
import Base.size #must import like this to add a definition to size

##Exports
export DataContainer, PairedDataContainer
export size, get_data, get_inputs, get_outputs, get_input_sample, get_output_sample, get_sample


## Objects
"""
    struct DataContainer

stores data samples as columns in an array
"""
struct DataContainer{FT <: Real}
    #stored data, each piece of data is a column [number of samples x data size]
    stored_data::Array{FT,2}    
    DataContainer(stored_data::Array{FT,2}; data_are_columns=true) where {FT <: Real} = data_are_columns ? new{FT}(stored_data) : new{FT}(permutedims(stored_data,(2,1)))
end

"""
    PairedDataContainer

stores input - output pairs as data containers
"""
struct PairedDataContainer{FT <: Real}
    #container for inputs and ouputs [sample_size x data/parameter size]
    inputs::DataContainer{FT} 
    outputs::DataContainer{FT}
    function PairedDataContainer(
        inputs::Array{FT,2},
        outputs::Array{FT,2};
        data_are_columns=true) where {FT <: Real}

        if data_are_columns
            if !(size(inputs,2) == size(outputs,2))
                throw(DimensionMismatch("There must be the same number of samples of both inputs and outputs"))
            else
                stored_inputs = DataContainer(inputs)
                stored_outputs = DataContainer(outputs)
                new{FT}(stored_inputs,stored_outputs)
            end
        else
            if !(size(inputs,1) == size(outputs,1))
                throw(DimensionMismatch("There must be the same number of samples of both inputs and outputs"))
            else
                stored_inputs = DataContainer{FT}(inputs; data_are_columns=false)
                stored_outputs = DataContainer{FT}(outputs; data_are_columns=false)
                new{FT}(inputs,outputs)
            end
        end
    end
    
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

function get_input_sample(pdc::PairedDataContainer,idx::IT) where {IT <: Integer}
    return get_sample(pdc.inputs,index)
end

function get_output_sample(pdc::PairedDataContainer,idx::IT) where {IT <: Integer}
    return get_sample(pdc.outputs,index)
end

function get_sample(pdc::PairedDataContainer, idx::IT) where {IT <: Integer}
    return get_input_sample(pdc,idx), get_output_sample(pdc,idx)
end

function get_sample(dc::DataContainer,idx::IT) where {IT <: Integer}
    return dc.stored_data[:,idx]
end

# Statistics? e.g get mean/cov/add_inflation could be done here.

end
