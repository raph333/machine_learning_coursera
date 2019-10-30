function [loss] = logloss(prediction, label)
loss = -(label * log(prediction)) - ((1-label) * (log(1-prediction)));
end