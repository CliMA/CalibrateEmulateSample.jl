function linkfig(figpath)
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://" * join(split(figpath, '/')[end-3:end], '/')
        alt = split(splitdir(figpath)[2], '.')[1]
        @info "Linking Figure: $artifact_url"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end