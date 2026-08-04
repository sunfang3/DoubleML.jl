module DoubleMLMLJExt

import DoubleML
import MLJBase

const MLJLearner = DoubleML.MLJLearner

"""
    MLJLearner(model; classifier=false, positive_label=1)

Adapt an MLJ model to the DoubleML nuisance-learner protocol. The adapter is
available only when `MLJBase` is loaded, so the core package remains usable
without the MLJ stack.

DoubleML owns the sample splits. A fresh MLJ machine is created for every
cross-fitting fold and is retained only when the caller requests models.
"""
DoubleML.is_classifier(m::MLJLearner) = m.classifier

function DoubleML.clone(m::MLJLearner)
    return MLJLearner(deepcopy(m.model);
                      classifier=m.classifier,
                      positive_label=m.positive_label)
end

function _features(X::AbstractMatrix)
    return MLJBase.table(Matrix{Float64}(X))
end

function _target(y::AbstractVector, classifier::Bool)
    return classifier ? MLJBase.categorical(y) : Float64.(y)
end

function DoubleML.fit!(m::MLJLearner, X::AbstractMatrix, y::AbstractVector)
    size(X, 1) == length(y) || throw(DimensionMismatch("X and y size mismatch"))
    m.machine = MLJBase.machine(m.model, _features(X), _target(y, m.classifier);
                                cache=false)
    MLJBase.fit!(m.machine; verbosity=0)
    m.fitted = true
    return m
end

function DoubleML.predict(m::MLJLearner, X::AbstractMatrix)
    m.fitted || error("MLJLearner is not fitted")
    m.classifier && return Float64.(MLJBase.predict_mode(m.machine, _features(X)) .== m.positive_label)
    return Float64.(MLJBase.predict_mean(m.machine, _features(X)))
end

function DoubleML.predict_proba(m::MLJLearner, X::AbstractMatrix)
    m.fitted || error("MLJLearner is not fitted")
    m.classifier || return DoubleML.predict(m, X)
    predictions = MLJBase.predict(m.machine, _features(X))
    return Float64[MLJBase.pdf(prediction, m.positive_label) for prediction in predictions]
end

function DoubleML.get_params(m::MLJLearner)
    params = MLJBase.params(m.model)
    return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(params))
end

function DoubleML.set_params!(m::MLJLearner; kwargs...)
    for (key, value) in pairs(kwargs)
        MLJBase.recursive_setproperty!(m.model, key, value)
    end
    m.machine = nothing
    m.fitted = false
    return m
end

end
