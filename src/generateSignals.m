function [] = generateSignals(frames_per_mod_type, MAX_PPM, Path, varargin) %Parameters (* = Required): *frames_per_mod_type, *MAX_PPM, *Path, SPF, SPS, HardSetPPM, freq_scale_factor ) 
    % Creating Degault values for parameters: spf, sps, hardsetppm, str_sq
    spf = 1024;
    sps = 8;
    set_ppm_offset = 1; 
    freq_scale_factor  = 1;    %% freq_scale_factor  = 1:  Increase Clock Frequency (Squeeze) ... freq_scale_factor  = -1: Decrease Clock Frequency (Stretch)
    
    fs = 200e3;             % Sample rate
    fc = [902e6 100e6];     % Center frequencies
    snr_levels = (0:2:30);
    int8_scale = 127;
    tx_delay = 50;
    file_name_root = 'frame';

    if(~endsWith(pwd(), 'RadioShift/src'))
        error("Please Run from Src in Radio Shift");
    end
    

    mod_types = categorical(["BPSK", "QPSK", "8PSK", ...
                                "16QAM","32QAM", "64QAM", "128QAM", "256QAM",...
                                "16APSK", "32APSK", "64APSK", "128APSK",...
                                "GFSK", "CPFSK", "FM", "AM-DSB-SC", "AM-SSB-SC"]);
                        
    num_mod_types = length(mod_types);

    defaults = {spf, sps, set_ppm_offset, freq_scale_factor };

    num_of_defaults = length(varargin);
    if(num_of_defaults > 0 )
        for i = 1:num_of_defaults
            defaults{i} = varargin{i};
        end
    end

    set_ppm = (freq_scale_factor  * MAX_PPM)/1e6';
    
    data_directory = fullfile(Path,"ModClassDataFiles");

    fprintf("Data file directory is %s \n", string(data_directory));

    % Check if data files exist

    if exist(data_directory,'dir')
        fprintf("%s exists and generated signals will be saved here\n", string(data_directory));
    else
        fprintf("Creating directory at: %s\n", string(data_directory));
        [success,msg,msgID] = mkdir(data_directory);
        if ~(success)
            error(msgID,msg)
        end
    end

    
    % data_directory_int8 = fullfile(data_directory, "int8");
    % data_directory_float32 = fullfile(data_directory, "float32");

    % if(exist(data_directory_int8, 'dir'))
    %     rmdir(data_directory_int8, 's');
    % end
    % if(exist(data_directory_float32, 'dir'))
    %     rmdir(data_directory_float32, 's')
    % end
                        
    % fprintf("Creating sub directory for float 32 at %s\n", string(data_directory_float32));
    % [success,msg,msgID] = mkdir(data_directory_float32);
    % if ~(success)
    %     error(msgID,msg)
    % end

    % fprintf("Creating sub directory for int8 at %s\n", string(data_directory_int8));
    % [success,msg,msgID] = mkdir(data_directory_int8);
    
    % if ~(success)
    %     error(msgID,msg)
    % end

    channel = dlhdlhelperModClassTestChannel(...
        'SampleRate', fs, ...
        'SNR', snr_levels(1), ...
        'PathDelays', [0 1.8 3.4] / fs, ...
        'AveragePathGains', [0 -2 -10], ...
        'KFactor', 4, ...
        'MaximumDopplerShift', 4, ...
        'MaximumClockOffset', set_ppm, ... %Set this as max PPM
        'HardSetOffsetPPM', set_ppm_offset,... % If this is set as 1, we hardset the Max PPM, otherwise we use a random PPM
        'CenterFrequency', fc(1));

    rng(1235)
    channel_info = info(channel);
    total_frame_count = num_mod_types*frames_per_mod_type*length(snr_levels);
    fprintf("A total of %d frames will be generated...\n",total_frame_count);
    all_IQ_int8 = cell(total_frame_count, 1);
    all_IQ_float32 = cell(total_frame_count, 1);
    all_labels = cell(total_frame_count, 1);
    all_SNRs = zeros(total_frame_count, 1);
    files_count_tracker = 1;
    for mod = 1:num_mod_types
        elapsed_time = seconds(toc);
        elapsed_time.Format = 'hh:mm:ss';
        fprintf('%s - Generating %s frames\n', ...
        elapsed_time, mod_types(mod))
        label = mod_types(mod);
        dataSrc = dlhdlhelperModClassGetSource(mod_types(mod), sps, 2*spf, fs);
        modulator = dlhdlhelperModClassGetModulator(mod_types(mod), sps, fs);
        if contains(char(mod_types(mod)), {'FM','AM-DSB-SC','AM-SSB-SC'})
        % Analog modulation types use a center frequency of 100 MHz
        channel.CenterFrequency = 100e6;
        else
        % Digital modulation types use a center frequency of 902 MHz
        channel.CenterFrequency = 902e6;
        end
        wb = waitbar(0,sprintf("Generating frames for %s...", label));
        for j = (1:length(snr_levels))
            for p=(1:frames_per_mod_type)
                % Generate random data
                x = dataSrc();
                SNR = snr_levels(j);
                
                % Modulate
                y = modulator(x);
                channel.SNR = SNR;
                % Pass through independent channels
                rx_samples = channel(y);
                
                % Remove transients from the beginning, trim to size, and normalize
                frame = dlhdlhelperModClassFrameGenerator(rx_samples, spf, spf, tx_delay, sps);
                
                % Save data file
                label = char(label);
                IQ = [real(frame),imag(frame)];
            
                % Saving Int 8 Dataset
                frame_IQ = int8(IQ * int8_scale);
                all_IQ_int8{files_count_tracker} = frame_IQ;
            
                % Saving Float32 Dataset
                frame_IQ = single(IQ);
                all_IQ_float32{files_count_tracker} = frame_IQ;

                all_labels{files_count_tracker} = label;
                all_SNRs(files_count_tracker) = SNR;

                files_count_tracker = files_count_tracker+1;
            end
            waitbar(j/length(snr_levels));
        end
        close(wb);
    end
    file_location = fullfile(data_directory,"MatGenData.mat");
    save(file_location, "all_IQ_int8", "all_IQ_float32", "all_labels", "all_SNRs", "-v7.3");
    fprintf("Saved the generated data at location %s", file_location);
end
