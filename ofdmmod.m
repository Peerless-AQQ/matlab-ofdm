clear; close all; clc;

%% 1. Параметры системы
Nfft = 1024;
Ng = round(Nfft / 4);
Nused = 840;

Mod_Order = 4;
M = 2^Mod_Order;

Fs = 20e6;

% Параметры канала
channel = comm.RayleighChannel(...
    'SampleRate', Fs, ...
    'PathDelays', [0 110 190 410] * 1e-9, ...
    'AveragePathGains', [0 -9.7 -19.2 -22.8], ...
    'NormalizePathGains', true, ...
    'MaximumDopplerShift', 5, ...
    'DopplerSpectrum', doppler('Jakes'), ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 42);

% Параметры для шага 16
pilot_spacing_16 = 16;
pilot_indices_16 = 1:pilot_spacing_16:Nused;
data_indices_16 = setdiff(1:Nused, pilot_indices_16);
Ndata_16 = length(data_indices_16);
Npilot_16 = length(pilot_indices_16);

% Параметры для шага 4
pilot_spacing_4 = 4;
pilot_indices_4 = 1:pilot_spacing_4:Nused;
data_indices_4 = setdiff(1:Nused, pilot_indices_4);
Ndata_4 = length(data_indices_4);
Npilot_4 = length(pilot_indices_4);

% SNR параметры
SNRdB = 0:5:50;
nSNR = length(SNRdB);

% Предварительное выделение массивов
BER_median_16 = zeros(size(SNRdB));
BER_linear_16 = zeros(size(SNRdB));
BER_spline_16 = zeros(size(SNRdB));
BER_median_4 = zeros(size(SNRdB));
BER_linear_4 = zeros(size(SNRdB));
BER_spline_4 = zeros(size(SNRdB));

BER_LS_linear_16 = zeros(size(SNRdB));
BER_MMSE_linear_16 = zeros(size(SNRdB));

MIN_ERRORS = 100;

rng(42);

fprintf('OFDM with MATLAB Rayleigh Channel\n');
fprintf('Modulation: %d-QAM\n', M);
fprintf('FFT size: %d, CP length: %d\n', Nfft, Ng);
fprintf('Used subcarriers: %d\n', Nused);
fprintf('Graph 1: LS with different interpolations (step 4 vs step 16)\n');
fprintf('Graph 2: LS vs MMSE with linear interpolation (step 16)\n');
fprintf('MMSE плавно вырождается в LS при SNR >= 25 dB\n\n');

%% Функция для теоретического BER
function ber = theoretical_ber_ofdm_rayleigh(EbN0_dB, M, Ndata, Nfft, delay_spread, Fs, num_paths)
    Bc = 1 / (delay_spread * 1e-9);
    B_sub = Fs / Nfft;
    Nc = round(Bc / B_sub);
    diversity_order = min(round(Ndata / Nc), num_paths);
    
    EbN0 = 10.^(EbN0_dB/10);
    k = log2(M);
    EsN0 = k * EbN0;
    
    L = sqrt(M);
    D = diversity_order;
    
    ber = zeros(size(EbN0_dB));
    
    for i = 1:length(EbN0_dB)
        gamma_s = EsN0(i);
        gamma_c = gamma_s / D;
        
        if D == 1
            ber(i) = 2*(1-1/L)/k * (1 - sqrt(1.5*gamma_s/(L^2-1 + 1.5*gamma_s)));
        else
            g_qam = 1.5 / (L^2 - 1);
            fun = @(theta) (4/k)*(1-1/L) * ...
                (1./(1 + g_qam*gamma_c./(sin(theta).^2))).^D;
            ber(i) = (1/pi) * integral(fun, 0, pi/2);
        end
    end
end

%% Функция LS оценки канала с разными интерполяциями
function H_est = estimate_channel_ls(Y_used, pilot_indices, pilot_syms, Nused, method)
    H_LS_at_pilots = Y_used(pilot_indices) ./ pilot_syms;
    
    switch lower(method)
        case 'linear'
            try
                H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'linear', 'extrap');
            catch
                H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'linear');
            end
            H_est = H_interp(:);
            
        case 'spline'
            try
                H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'spline', 'extrap');
            catch
                try
                    H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'pchip', 'extrap');
                catch
                    H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'linear', 'extrap');
                end
            end
            H_est = H_interp(:);
            
        case 'median'
            try
                H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'spline', 'extrap');
            catch
                try
                    H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'pchip', 'extrap');
                catch
                    H_interp = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'linear', 'extrap');
                end
            end
            H_interp = H_interp(:);
            
            if any(isnan(H_interp))
                H_interp(isnan(H_interp)) = mean(H_interp(~isnan(H_interp)));
            end
            
            window_size = min(7, floor(length(H_interp)/4));
            if window_size < 3
                window_size = 3;
            end
            
            try
                H_est = medfilt1(real(H_interp), window_size) + ...
                        1j * medfilt1(imag(H_interp), window_size);
            catch
                H_est = H_interp;
            end
            
        otherwise
            error('Unknown method: %s', method);
    end
    
    if any(isnan(H_est)) || any(isinf(H_est))
        H_est(isnan(H_est) | isinf(H_est)) = 1;
    end
end

%% Функция MMSE оценки канала (с ПЛАВНЫМ вырождением в LS)
function H_est = mmse_channel_estimate(Y_used, pilot_indices, pilot_syms, Nused, SNR_linear, Nfft, delay_spread_ns, Fs)
    H_LS_at_pilots = Y_used(pilot_indices) ./ pilot_syms;
    Np = length(pilot_indices);
    
    % Вычисляем LS оценку (всегда)
    try
        H_ls = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'linear', 'extrap');
    catch
        H_ls = interp1(pilot_indices, H_LS_at_pilots, 1:Nused, 'linear');
    end
    H_ls = H_ls(:);
    
    % Плавный коэффициент вырождения
    SNR_dB = 10*log10(max(SNR_linear, eps));
    
    % Сигмоида: alpha = 0 при низком SNR (MMSE), alpha = 1 при высоком SNR (LS)
    % Центр перехода: 27.5 дБ, ширина перехода: ~5 дБ
    alpha = 1 / (1 + exp(-(SNR_dB - 27.5) / 2.5));
    
    % Для очень низких SNR: alpha ≈ 0 → чистый MMSE
    % Для очень высоких SNR: alpha ≈ 1 → чистый LS
    
    if alpha > 0.999
        % Практически полное вырождение → используем LS
        H_est = H_ls;
        return;
    end
    
    if alpha < 0.001
        % Практически чистый MMSE
        use_pure_mmse = true;
    else
        use_pure_mmse = false;
    end
    
    % Вычисляем MMSE оценку (если нужно)
    if ~use_pure_mmse || alpha < 0.5
        tau_rms = delay_spread_ns * 1e-9 / (2*pi);
        subcarrier_spacing = Fs / Nfft;
        freq_diff = abs((1:Nused)' - (1:Nused)) * subcarrier_spacing;
        
        R_HH = 1 ./ (1 + (2*pi*tau_rms*freq_diff).^2);
        noise_power = 1 / SNR_linear;
        
        R_HpHp = R_HH(pilot_indices, pilot_indices);
        R_hHp = R_HH(:, pilot_indices);
        R_HpHp_reg = R_HpHp + noise_power * eye(Np);
        
        try
            W = R_hHp / R_HpHp_reg;
            H_mmse = W * H_LS_at_pilots(:);
        catch
            H_mmse = H_ls;
        end
    else
        H_mmse = H_ls;
    end
    
    % Плавное смешивание LS и MMSE
    H_est = alpha * H_ls + (1 - alpha) * H_mmse;
    
    if any(isnan(H_est)) || any(isinf(H_est))
        H_est = H_ls;
    end
end

%% ======================== СИМУЛЯЦИЯ: ГРАФИК 1 ========================
fprintf('\n========== ГРАФИК 1: LS с разными интерполяциями ==========\n');

for i_snr = 1:nSNR
    
    EsN0_target_linear = 10^(SNRdB(i_snr)/10);
    
    err_median_16 = 0; err_linear_16 = 0; err_spline_16 = 0;
    err_median_4 = 0; err_linear_4 = 0; err_spline_4 = 0;
    
    total_bits_16 = 0;
    total_bits_4 = 0;
    
    if SNRdB(i_snr) >= 45
        Target_Bits = 50e6; Max_Bits = 200e6;
    elseif SNRdB(i_snr) >= 40
        Target_Bits = 20e6; Max_Bits = 80e6;
    elseif SNRdB(i_snr) >= 35
        Target_Bits = 10e6; Max_Bits = 50e6;
    else
        Target_Bits = 1e6; Max_Bits = 10e6;
    end
    
    fprintf('SNR = %5.1f dB: ', SNRdB(i_snr));
    tic;
    
    while (total_bits_16 < Target_Bits && total_bits_4 < Target_Bits) || ...
          (min([err_median_16, err_linear_16, err_spline_16, ...
                err_median_4, err_linear_4, err_spline_4]) < MIN_ERRORS && ...
           total_bits_16 < Max_Bits)
        
        %% Общий передатчик
        all_data_syms = qammod(randi([0 1], Nused * Mod_Order, 1), M, ...
            'InputType', 'bit', 'UnitAveragePower', true);
        
        freq_syms = all_data_syms;
        
        X = zeros(Nfft, 1);
        used_left = 2:(Nused/2 + 1);
        used_right = Nfft - Nused/2 + 1 : Nfft;
        used_all = [used_left, used_right];
        X(used_all) = freq_syms;
        
        x_time = ifft(ifftshift(X)) * sqrt(Nfft);
        x_cp = [x_time(end-Ng+1:end); x_time];
        
        %% КАНАЛ
        reset(channel);
        y_channel = channel(complex(x_cp));
        
        %% ШУМ
        noise = sqrt(1/(2*EsN0_target_linear)) * (randn(size(y_channel)) + 1j*randn(size(y_channel)));
        y_rx = y_channel + noise;
        
        %% ПРИЁМНИК
        y_time = y_rx(Ng+1:end);
        Y_freq = fftshift(fft(y_time)) / sqrt(Nfft);
        Y_used = Y_freq(used_all);
        
        %% ===== Шаг 16 =====
        pilot_syms_16 = all_data_syms(pilot_indices_16);
        tx_data_bits_16 = qamdemod(all_data_syms(data_indices_16), M, ...
            'OutputType', 'bit', 'UnitAveragePower', true);
        total_bits_16 = total_bits_16 + length(tx_data_bits_16);
        
        H_est_median_16 = estimate_channel_ls(Y_used, pilot_indices_16, pilot_syms_16, Nused, 'median');
        H_est_linear_16 = estimate_channel_ls(Y_used, pilot_indices_16, pilot_syms_16, Nused, 'linear');
        H_est_spline_16 = estimate_channel_ls(Y_used, pilot_indices_16, pilot_syms_16, Nused, 'spline');
        
        Y_data_16 = Y_used(data_indices_16);
        
        % Linear 16
        H_data = H_est_linear_16(data_indices_16);
        X_eq = conj(H_data) .* Y_data_16 ./ (abs(H_data).^2 + 1/EsN0_target_linear + eps);
        rx_bits = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_linear_16 = err_linear_16 + sum(tx_data_bits_16 ~= rx_bits);
        
        % Median 16
        H_data = H_est_median_16(data_indices_16);
        X_eq = conj(H_data) .* Y_data_16 ./ (abs(H_data).^2 + 1/EsN0_target_linear + eps);
        rx_bits = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_median_16 = err_median_16 + sum(tx_data_bits_16 ~= rx_bits);
        
        % Spline 16
        H_data = H_est_spline_16(data_indices_16);
        X_eq = conj(H_data) .* Y_data_16 ./ (abs(H_data).^2 + 1/EsN0_target_linear + eps);
        rx_bits = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_spline_16 = err_spline_16 + sum(tx_data_bits_16 ~= rx_bits);
        
        %% ===== Шаг 4 =====
        pilot_syms_4 = all_data_syms(pilot_indices_4);
        tx_data_bits_4 = qamdemod(all_data_syms(data_indices_4), M, ...
            'OutputType', 'bit', 'UnitAveragePower', true);
        total_bits_4 = total_bits_4 + length(tx_data_bits_4);
        
        H_est_median_4 = estimate_channel_ls(Y_used, pilot_indices_4, pilot_syms_4, Nused, 'median');
        H_est_linear_4 = estimate_channel_ls(Y_used, pilot_indices_4, pilot_syms_4, Nused, 'linear');
        H_est_spline_4 = estimate_channel_ls(Y_used, pilot_indices_4, pilot_syms_4, Nused, 'spline');
        
        Y_data_4 = Y_used(data_indices_4);
        
        % Linear 4
        H_data = H_est_linear_4(data_indices_4);
        X_eq = conj(H_data) .* Y_data_4 ./ (abs(H_data).^2 + 1/EsN0_target_linear + eps);
        rx_bits = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_linear_4 = err_linear_4 + sum(tx_data_bits_4 ~= rx_bits);
        
        % Median 4
        H_data = H_est_median_4(data_indices_4);
        X_eq = conj(H_data) .* Y_data_4 ./ (abs(H_data).^2 + 1/EsN0_target_linear + eps);
        rx_bits = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_median_4 = err_median_4 + sum(tx_data_bits_4 ~= rx_bits);
        
        % Spline 4
        H_data = H_est_spline_4(data_indices_4);
        X_eq = conj(H_data) .* Y_data_4 ./ (abs(H_data).^2 + 1/EsN0_target_linear + eps);
        rx_bits = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_spline_4 = err_spline_4 + sum(tx_data_bits_4 ~= rx_bits);
        
        if mod(total_bits_16, 500000) < Ndata_16 * Mod_Order
            fprintf('.');
        end
    end
    
    BER_median_16(i_snr) = err_median_16 / total_bits_16;
    BER_linear_16(i_snr) = err_linear_16 / total_bits_16;
    BER_spline_16(i_snr) = err_spline_16 / total_bits_16;
    
    BER_median_4(i_snr) = err_median_4 / total_bits_4;
    BER_linear_4(i_snr) = err_linear_4 / total_bits_4;
    BER_spline_4(i_snr) = err_spline_4 / total_bits_4;
    
    elapsed_time = toc;
    fprintf(' Time: %.1f s\n', elapsed_time);
    fprintf('  Step 16 — Median: %.3e | Linear: %.3e | Spline: %.3e\n', ...
        BER_median_16(i_snr), BER_linear_16(i_snr), BER_spline_16(i_snr));
    fprintf('  Step 4  — Median: %.3e | Linear: %.3e | Spline: %.3e\n', ...
        BER_median_4(i_snr), BER_linear_4(i_snr), BER_spline_4(i_snr));
end

%% ======================== СИМУЛЯЦИЯ: ГРАФИК 2 (шаг 16, LS vs MMSE) ========================
fprintf('\n========== ГРАФИК 2: LS vs MMSE (шаг 16) ==========\n');

rng(123);
delay_spread_ns = max([0 110 190 410]);

for i_snr = 1:nSNR
    
    EsN0_target_linear = 10^(SNRdB(i_snr)/10);
    
    err_ls = 0;
    err_mmse = 0;
    total_bits = 0;
    
    if SNRdB(i_snr) >= 45
        Target_Bits = 50e6; Max_Bits = 200e6;
    elseif SNRdB(i_snr) >= 40
        Target_Bits = 20e6; Max_Bits = 80e6;
    elseif SNRdB(i_snr) >= 35
        Target_Bits = 10e6; Max_Bits = 50e6;
    else
        Target_Bits = 2e6; Max_Bits = 10e6;
    end
    
    fprintf('SNR = %5.1f dB: ', SNRdB(i_snr));
    tic;
    
    while total_bits < Target_Bits || ...
          (min([err_ls, err_mmse]) < MIN_ERRORS && total_bits < Max_Bits)
        
        %% ПЕРЕДАТЧИК
        tx_data_bits = randi([0 1], Ndata_16 * Mod_Order, 1);
        total_bits = total_bits + length(tx_data_bits);
        
        data_syms = qammod(tx_data_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
        
        pilot_bits = randi([0 1], Npilot_16 * Mod_Order, 1);
        pilot_syms = qammod(pilot_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
        
        freq_syms = zeros(Nused, 1);
        freq_syms(data_indices_16) = data_syms;
        freq_syms(pilot_indices_16) = pilot_syms;
        
        X = zeros(Nfft, 1);
        used_left = 2:(Nused/2 + 1);
        used_right = Nfft - Nused/2 + 1 : Nfft;
        used_all = [used_left, used_right];
        X(used_all) = freq_syms;
        
        x_time = ifft(ifftshift(X)) * sqrt(Nfft);
        x_cp = [x_time(end-Ng+1:end); x_time];
        
        %% КАНАЛ
        reset(channel);
        y_channel = channel(complex(x_cp));
        
        %% ШУМ
        noise = sqrt(1/(2*EsN0_target_linear)) * (randn(size(y_channel)) + 1j*randn(size(y_channel)));
        y_rx = y_channel + noise;
        
        %% ПРИЁМНИК
        y_time = y_rx(Ng+1:end);
        Y_freq = fftshift(fft(y_time)) / sqrt(Nfft);
        Y_used = Y_freq(used_all);
        
        %% LS + Linear
        H_est_ls = estimate_channel_ls(Y_used, pilot_indices_16, pilot_syms, Nused, 'linear');
        H_data_ls = H_est_ls(data_indices_16);
        X_eq_ls = conj(H_data_ls) .* Y_used(data_indices_16) ./ (abs(H_data_ls).^2 + 1/EsN0_target_linear + eps);
        rx_bits_ls = qamdemod(X_eq_ls, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_ls = err_ls + sum(tx_data_bits ~= rx_bits_ls);
        
        %% MMSE + Linear (с ПЛАВНЫМ вырождением в LS)
        H_est_mmse = mmse_channel_estimate(Y_used, pilot_indices_16, pilot_syms, ...
            Nused, EsN0_target_linear, Nfft, delay_spread_ns, Fs);
        H_data_mmse = H_est_mmse(data_indices_16);
        X_eq_mmse = conj(H_data_mmse) .* Y_used(data_indices_16) ./ (abs(H_data_mmse).^2 + 1/EsN0_target_linear + eps);
        rx_bits_mmse = qamdemod(X_eq_mmse, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_mmse = err_mmse + sum(tx_data_bits ~= rx_bits_mmse);
        
        if mod(total_bits, 500000) < Ndata_16 * Mod_Order
            fprintf('.');
        end
        
        if total_bits >= Max_Bits && min([err_ls, err_mmse]) < MIN_ERRORS
            break;
        end
    end
    
    BER_LS_linear_16(i_snr) = err_ls / total_bits;
    BER_MMSE_linear_16(i_snr) = err_mmse / total_bits;
    
    elapsed_time = toc;
    fprintf(' Time: %.1f s\n', elapsed_time);
    fprintf('  LS+Linear: %.3e | MMSE+Linear: %.3e', ...
        BER_LS_linear_16(i_snr), BER_MMSE_linear_16(i_snr));
    
    if err_ls > 0 && err_mmse > 0
        gain_dB = 10*log10(BER_LS_linear_16(i_snr) / BER_MMSE_linear_16(i_snr));
        fprintf(' | Gain: %.1f dB\n', gain_dB);
    else
        fprintf('\n');
    end
end

%% ======================== ВИЗУАЛИЗАЦИЯ ========================

num_paths = 4;
EbN0_dB = SNRdB - 10*log10(Mod_Order);

BER_ideal_16 = theoretical_ber_ofdm_rayleigh(EbN0_dB, M, Ndata_16, Nfft, delay_spread_ns, Fs, num_paths);
BER_ideal_4 = theoretical_ber_ofdm_rayleigh(EbN0_dB, M, Ndata_4, Nfft, delay_spread_ns, Fs, num_paths);

fprintf('\nDiversity order: %d (limited by %d paths)\n', num_paths, num_paths);

%% ГРАФИК 1: LS с разными интерполяциями и шагом пилотов
figure('Position', [100, 100, 1000, 700]);

semilogy(SNRdB, BER_ideal_16, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Идеальный CSI');
hold on;

colors = {'b', 'r', 'g'};
markers = {'o', 's', '^'};
methods = {'Median', 'Linear', 'Spline'};

ber_data_16 = {BER_median_16, BER_linear_16, BER_spline_16};
for m = 1:3
    valid = ber_data_16{m} > 0;
    if any(valid)
        semilogy(SNRdB(valid), ber_data_16{m}(valid), [colors{m} '-' markers{m}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', colors{m}, ...
            'DisplayName', ['Шаг 16: ' methods{m}]);
    end
end

ber_data_4 = {BER_median_4, BER_linear_4, BER_spline_4};
for m = 1:3
    valid = ber_data_4{m} > 0;
    if any(valid)
        semilogy(SNRdB(valid), ber_data_4{m}(valid), [colors{m} '--' markers{m}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
            'DisplayName', ['Шаг 4: ' methods{m}]);
    end
end

grid on;
xlabel('SNR на поднесущую, Es/N0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title('График 1: LS оценка с разными интерполяциями', 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 8);
axis([min(SNRdB) max(SNRdB) 1e-6 1]);

text(min(SNRdB)+1, 2e-6, ...
    sprintf(['― : шаг 16 (%d пилотов, %d данных)\n', ...
             '- - : шаг 4 (%d пилотов, %d данных)\n', ...
             'Diversity order = %d (ограничен %d лучами)'], ...
             Npilot_16, Ndata_16, Npilot_4, Ndata_4, num_paths, num_paths), ...
    'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

%% ГРАФИК 2: LS vs MMSE (шаг 16, с плавным вырождением)
figure('Position', [150, 150, 900, 650]);

semilogy(SNRdB, BER_ideal_16, 'k-', 'LineWidth', 2.5, ...
    'DisplayName', 'Идеальный CSI (теория)');
hold on;

valid_ls = BER_LS_linear_16 > 0;
valid_mmse = BER_MMSE_linear_16 > 0;

if any(valid_ls)
    semilogy(SNRdB(valid_ls), BER_LS_linear_16(valid_ls), 'b-s', ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b', ...
        'DisplayName', 'LS + Линейная интерполяция');
end

if any(valid_mmse)
    semilogy(SNRdB(valid_mmse), BER_MMSE_linear_16(valid_mmse), 'r-o', ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
        'DisplayName', 'MMSE (→ LS при SNR >= 25 dB)');
end

grid on;
xlabel('SNR на поднесущую, Es/N0 (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title('График 2: LS vs MMSE (шаг 16, плавное вырождение)', 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 10);
axis([min(SNRdB) max(SNRdB) 1e-6 1]);

text(min(SNRdB)+1, 2e-6, ...
    sprintf(['Шаг пилотов: %d (%d пилотов, %d данных)\n', ...
             'MMSE → LS: сигмоида (центр 27.5 dB, ширина 5 dB)\n', ...
             '16-QAM, Rayleigh (410 нс), Diversity = %d'], ...
             pilot_spacing_16, Npilot_16, Ndata_16, num_paths), ...
    'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

%% СВОДНАЯ ТАБЛИЦА
fprintf('\n=== РЕЗУЛЬТАТЫ: LS vs MMSE (шаг 16) ===\n');
fprintf('SNR(dB) | LS+Linear | MMSE+Linear | Gain(dB) | Лучше\n');
fprintf('--------|-----------|-------------|----------|------\n');
for i = 1:nSNR
    if BER_LS_linear_16(i) > 0 && BER_MMSE_linear_16(i) > 0
        gain_dB = 10*log10(BER_LS_linear_16(i) / BER_MMSE_linear_16(i));
        if gain_dB > 0.05
            winner = 'MMSE';
        elseif gain_dB < -0.05
            winner = 'LS';
        else
            winner = '≈ (вырождение)';
        end
    else
        gain_dB = 0;
        winner = '?';
    end
    
    fprintf('%7.1f | %9.3e | %11.3e | %8.1f | %s\n', ...
        SNRdB(i), BER_LS_linear_16(i), BER_MMSE_linear_16(i), gain_dB, winner);
end

% Подсчёт побед
wins_ls = sum(BER_LS_linear_16 < BER_MMSE_linear_16 & BER_LS_linear_16 > 0);
wins_mmse = sum(BER_MMSE_linear_16 < BER_LS_linear_16 & BER_MMSE_linear_16 > 0);
fprintf('\nLS победил: %d раз\n', wins_ls);
fprintf('MMSE победил: %d раз\n', wins_mmse);

fprintf('\nГотово!\n');