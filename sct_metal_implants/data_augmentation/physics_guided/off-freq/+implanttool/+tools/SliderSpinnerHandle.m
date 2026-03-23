% A handle linking a slider and a spinner
classdef SliderSpinnerHandle < handle
    properties
        slider matlab.ui.control.Slider;
        spinner matlab.ui.control.Spinner;
        value;
        callback function_handle = @(obj) disp('');
    end
    methods
        function obj = SliderSpinnerHandle(slider, spinner, value, limits, step)
            arguments
                slider matlab.ui.control.Slider;
                spinner matlab.ui.control.Spinner;
                value = 0;
                limits = -1;
                step = -1;
            end
            obj.slider = slider;
            obj.spinner = spinner;

            if limits ~= -1
                value = max(value,limits(1));
                value = min(value,limits(2));
                obj.updateLimits(limits)
            end

            if step ~= -1
                obj.updateStep(step)
            end
            
            obj.updateCallbacks();
            obj.updateValue(value);
        end

        function [] = updateValue(obj, value)
            obj.value = value;
            obj.slider.Value = value;
            obj.spinner.Value = value;
            obj.callback()
        end

        function [] = updateLimits(obj, limits)
            obj.slider.Limits = limits;
            obj.spinner.Limits = limits;
        end

        function [] = updateStep(obj, step)
            obj.spinner.Step = step;
        end

        function [] = updateCallbacks(obj)
            obj.slider.ValueChangedFcn = @(es, ed) obj.updateValue(es.Value);
            obj.slider.ValueChangingFcn = @(es, ed) obj.updateValue(ed.Value);
            obj.spinner.ValueChangedFcn = @(es, ed) obj.updateValue(es.Value);
        end
    end
end
