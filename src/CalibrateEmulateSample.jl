"""
# Imported modules:
$(IMPORTS)

# Exports:
$(EXPORTS)
"""
module CalibrateEmulateSample

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# imported modules from EKP.
import EnsembleKalmanProcesses: EnsembleKalmanProcesses, ParameterDistributions, Observations, DataContainers

export EnsembleKalmanProcesses, ParameterDistributions, Observations, DataContainers

########################
########################
# bugfix for ProgressBars
# the only difference is the key "9" in the dict EIGHTS, following solve by:
# https://github.com/cloud-oak/ProgressBars.jl/issues/51
import ProgressBars
using Printf

EIGHTS = Dict(0 => ' ', 1 => '▏', 2 => '▎', 3 => '▍', 4 => '▌', 5 => '▋', 6 => '▊', 7 => '▉', 8 => '█', 9 => '█')

function ProgressBars.update(iter::ProgressBars.ProgressBar, amount::Int64 = 1; force_print = false)
    lock(iter.count_lock)
    current = try
        iter.current += amount
    finally
        unlock(iter.count_lock)
    end

    if current < iter.current_printed
        return
    end

    if current == iter.total
        if iter.leave
            force_print = true
        else
            ProgressBars.clear_progress(iter)
            return
        end
    end

    if force_print
        lock(iter.print_lock)
    elseif time_ns() - iter.last_print >= iter.printing_delay
        if !trylock(iter.print_lock)
            return
        end
    else
        return
    end

    try
        if !iter.fixwidth
            current_terminal_width = displaysize(iter.output_stream)[2]
            terminal_width_changed = current_terminal_width != iter.width
            if terminal_width_changed
                iter.width = current_terminal_width
                ProgressBars.make_space_after_progress_bar(iter.output_stream, iter.extra_lines)
            end
        end

        seconds = (time_ns() - iter.start_time) * 1e-9
        iteration = current - 1

        elapsed = ProgressBars.format_time(seconds)
        speed = iteration / seconds
        if seconds == 0
            # Dummy value of 1 it/s if no time has elapsed
            speed = 1
        end

        if speed >= 1
            iterations_per_second = ProgressBars.format_amount(speed, "$(iter.iter_unit)/s", iter.unit_scale)
        else
            # TODO: This might fail if speed == 0
            iterations_per_second = ProgressBars.format_amount(1 / speed, "s/$(iter.iter_unit)", iter.unit_scale)
        end

        barwidth = iter.width - 2 # minus two for the separators

        postfix_string = ProgressBars.postfix_repr(iter.postfix)

        # Reset Cursor to beginning of the line
        for line in 1:(iter.extra_lines)
            ProgressBars.move_up_1_line(iter.output_stream)
        end
        ProgressBars.go_to_start_of_line(iter.output_stream)

        if iter.description != ""
            barwidth -= length(iter.description) + 1
            print(iter.output_stream, iter.description * " ")
        end

        if (iter.total <= 0)
            current_string = ProgressBars.format_amount(iter.current[], iter.iter_unit, iter.unit_scale)
            status_string = "$(current_string) $elapsed [$iterations_per_second$postfix_string]"
            barwidth -= length(status_string) + 1
            if barwidth < 0
                barwidth = 0
            end

            print(iter.output_stream, "┣")
            print(iter.output_stream, join(IDLE[1 + ((i + current) % length(IDLE))] for i in 1:barwidth))
            print(iter.output_stream, "┫ ")
            print(iter.output_stream, status_string)
        else
            ETA = (iter.total - current) / speed

            percentage_string = string(@sprintf("%.1f%%", current / iter.total * 100))

            eta = ProgressBars.format_time(ETA)
            current_string = ProgressBars.format_amount(current, iter.unit, iter.unit_scale)
            total = ProgressBars.format_amount(iter.total, iter.unit, iter.unit_scale)
            status_string = "$(current_string)/$(total) [$elapsed<$eta, $iterations_per_second$postfix_string]"

            barwidth -= length(status_string) + length(percentage_string) + 1
            if barwidth < 0
                barwidth = 0
            end

            cellvalue = iter.total / barwidth
            full_cells, remain = divrem(current, cellvalue)

            print(iter.output_stream, percentage_string)
            print(iter.output_stream, "┣")
            print(iter.output_stream, repeat("█", Int(full_cells)))
            if (full_cells < barwidth)
                part = Int(floor(9 * remain / cellvalue))
                print(iter.output_stream, EIGHTS[part])
                print(iter.output_stream, repeat(" ", Int(barwidth - full_cells - 1)))
            end

            print(iter.output_stream, "┫ ")
            print(iter.output_stream, status_string)
        end
        multiline_postfix_string, iter.extra_lines = ProgressBars.newline_to_spaces(iter.multilinepostfix, iter.width)
        print(iter.output_stream, multiline_postfix_string)
        println(iter.output_stream)

        iter.last_print = time_ns()
        iter.current_printed = current
    finally
        unlock(iter.print_lock)
    end
end
################
################

# Internal deps, light external deps
include("Utilities.jl")

# No internal deps, heavy external deps
#include("GaussianProcessEmulator.jl")
include("Emulator.jl")

# Internal deps, light external deps
include("MarkovChainMonteCarlo.jl")

end # module
