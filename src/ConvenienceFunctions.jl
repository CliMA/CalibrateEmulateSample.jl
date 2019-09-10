
const RPAD = 25

function name(name::AbstractString)
  return rpad(name * ":", RPAD)
end

function warn(name::AbstractString)
  return rpad("WARNING (" * name * "):", RPAD)
end


