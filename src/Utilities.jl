module Utilities

export RPAD, name, warn

const RPAD = 25

name(_name::AbstractString) = rpad(_name * ":", RPAD)

warn(_name::AbstractString) = rpad("WARNING (" * _name * "):", RPAD)

end