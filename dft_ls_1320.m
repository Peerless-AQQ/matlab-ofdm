clear; close all; clc;

%% ПАРАМЕТРЫ
Nused = 1320;
Sf = 6;
Fs = 30.72e6;
Delta_f = 15e3;

% Comb-массив с расширением
N_ext = 6;
Ncomb_ext = Nused + 2*N_ext;
Ng_local = round(144 * Ncomb_ext / 2048);

used_start = N_ext + 1;
used_end   = N_ext + Nused;

pilot_in_used = (1:Sf:Nused).';
Npilot = length(pilot_in_used);
pilot_in_ext = pilot_in_used + N_ext;
data_in_used = setdiff((1:Nused).', pilot_in_used);
data_in_ext = data_in_used + N_ext;

% Веса линейной интерполяции
n_vec = (1:Nused).';
i_left = floor((n_vec - 1) / Sf) + 1;
i_left = min(i_left,  Npilot);
i_right = min(i_left + 1, Npilot);
pos_l = pilot_in_used(i_left);
pos_r = pilot_in_used(i_right);
denom = pos_r - pos_l;
denom(denom == 0) = 1;
t_vec = (n_vec - pos_l) ./ denom;
t_vec(pos_l == pos_r) = 0;
w_r = t_vec;
w_l = 1 - t_vec;

% Канал 
v_kmh = 30; fc = 4e9;
c = physconst('lightspeed');
fd = (v_kmh*1000/3600)/c*fc;
delay_spread_ns = 100;

tdl = nrTDLChannel;
tdl.DelayProfile        = 'TDL-E';
tdl.DelaySpread         = delay_spread_ns * 1e-9;
tdl.MaximumDopplerShift = fd;
tdl.SampleRate          = Fs;
tdl.NumTransmitAntennas = 1;
tdl.NumReceiveAntennas  = 1;
tdl.NormalizePathGains  = true;

info_tdl = info(tdl);
max_delay_s = max(info_tdl.PathDelays);

Fs_comb = Ncomb_ext * Delta_f;
T_rep = floor(Ncomb_ext / Sf);
max_delay_comb_samples = ceil(max_delay_s * Fs_comb);

% Параметры адаптивного усечения
k_noise_trunc = 5;
margin_trunc = 4;
L_max_trunc = max_delay_comb_samples * 2;

% Фиксированный L_trunc для сравнения
L_trunc_fixed = max_delay_comb_samples + 16;

fprintf('=== Адаптивное усечение, TDL-E ===\n');
fprintf('k_noise=%d, margin=%d, L_max=%d\n', ...
        k_noise_trunc, margin_trunc, L_max_trunc);
fprintf('Фикс. L_trunc = %d\n\n', L_trunc_fixed);

M = 16; Mod_Order = 4;
K_dB = 22;

%% ФУНКЦИИ
function ber = theoretical_ber_tdl_e(EbN0_dB, M, K_dB)
    k = log2(M); L = sqrt(M);
    EbN0 = 10.^(EbN0_dB/10);
    ber = (2/k) * (1 - 1/L) * erfc(sqrt(3 * EbN0 * k / (2 * (M - 1))));
    K = 10^(K_dB/10);
    ber = ber / (1 + 1/K);
end

function [h_trunc, L_trunc] = adaptive_truncation(h_comb, Sf, Ncomb_ext, ...
                                                   k_noise, margin, L_max)
    T_rep = floor(Ncomb_ext / Sf);
    h_one_copy = h_comb(1:T_rep);

    noise_tail = h_one_copy(round(T_rep/2):end-1);
    sigma_noise = sqrt(mean(abs(noise_tail).^2));

    if sigma_noise < eps
        L_trunc = 1;
    else
        threshold = k_noise * sigma_noise;
        significant = find(abs(h_one_copy) > threshold);
        if isempty(significant)
            L_last = 1;
        else
            L_last = significant(end);
        end
        L_trunc = min(L_last + margin, min(L_max, T_rep - 1));
    end

    L_trunc = max(L_trunc, 1);

    h_trunc = zeros(Ncomb_ext, 1);
    h_trunc(1:L_trunc) = h_one_copy(1:L_trunc) * Sf;
end

% BER-кривые
SNRdB = 0:5:25;
nSNR = length(SNRdB);
BER_comb_adaptive = zeros(size(SNRdB));
BER_comb_fixed = zeros(size(SNRdB));
BER_ls = zeros(size(SNRdB));

MIN_ERRORS   = 100;
MAX_BITS_CAP = 5e7;
rng(42);

BER_pred = theoretical_ber_tdl_e(SNRdB - 10*log10(Mod_Order), M, K_dB);

for i_snr = 1:nSNR
    EsN0_lin = 10^(SNRdB(i_snr)/10);
    err_comb_ad = 0; err_comb_fx = 0; err_ls = 0; total_bits = 0;

    fprintf('SNR = %5.1f дБ: ', SNRdB(i_snr));
    tic;

    while (err_comb_ad < MIN_ERRORS || err_comb_fx < MIN_ERRORS || ...
           err_ls < MIN_ERRORS) && total_bits < MAX_BITS_CAP

        data_bits = randi([0 1], length(data_in_used) * Mod_Order, 1);
        data_syms = qammod(data_bits, M, 'InputType', 'bit', ...
                           'UnitAveragePower', true);
        pilot_bits = randi([0 1], Npilot * Mod_Order, 1);
        pilot_syms = qammod(pilot_bits, M, 'InputType', 'bit', ...
                            'UnitAveragePower', true);

        freq_syms = zeros(Ncomb_ext, 1);
        freq_syms(data_in_ext)  = data_syms;
        freq_syms(pilot_in_ext) = pilot_syms;

        x_time = ifft(freq_syms) * sqrt(Ncomb_ext);
        x_cp   = [x_time(end-Ng_local+1:end); x_time];

        reset(tdl);
        y_ch = tdl(complex(x_cp));
        noise = sqrt(1/(2*EsN0_lin)) * ...
                (randn(size(y_ch)) + 1j*randn(size(y_ch)));
        y_rx = y_ch + noise;

        y_time = y_rx(Ng_local+1:end);
        Y_nat  = fft(y_time) / sqrt(Ncomb_ext);

        pilot_syms_nat = freq_syms(pilot_in_ext);
        H_LS_pil = Y_nat(pilot_in_ext) ./ pilot_syms_nat;

        if any(~isfinite(H_LS_pil))
            continue;
        end

        H_comb = zeros(Ncomb_ext, 1);
        H_comb(pilot_in_ext) = H_LS_pil;
        slope_L = H_LS_pil(2) - H_LS_pil(1);
        H_comb(1) = H_LS_pil(1) - slope_L;
        slope_R = H_LS_pil(end) - H_LS_pil(end-1);
        H_comb(Ncomb_ext - 5) = H_LS_pil(end) + slope_R;

        h_comb = ifft(H_comb) * sqrt(Ncomb_ext);

        % Адаптивное усечение
        [h_trunc_ad, ~] = adaptive_truncation(h_comb, Sf, Ncomb_ext, ...
                                              k_noise_trunc, ...
                                              margin_trunc, L_max_trunc);
        H_est_full_ad = fft(h_trunc_ad) / sqrt(Ncomb_ext);
        H_est_comb_ad = H_est_full_ad(used_start:used_end);

        % Фиксированное усечение
        h_trunc_fx = zeros(Ncomb_ext, 1);
        h_trunc_fx(1:L_trunc_fixed) = h_comb(1:L_trunc_fixed) * Sf;
        H_est_full_fx = fft(h_trunc_fx) / sqrt(Ncomb_ext);
        H_est_comb_fx = H_est_full_fx(used_start:used_end);

        % LS + linear
        H_est_ls = w_l .* H_LS_pil(i_left) + w_r .* H_LS_pil(i_right);

        % Эквалайзер
        Y_data_nat  = Y_nat(data_in_ext);
        tx_syms_nat = freq_syms(data_in_ext);
        tx_bits_used = qamdemod(tx_syms_nat, M, 'OutputType', 'bit', ...
                                'UnitAveragePower', true);

        H_d = H_est_comb_ad(data_in_used);
        X_eq = conj(H_d) .* Y_data_nat ./ (abs(H_d).^2 + 1/EsN0_lin + eps);
        X_eq(~isfinite(X_eq)) = 0;
        rx = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_comb_ad = err_comb_ad + sum(tx_bits_used ~= rx);

        H_d = H_est_comb_fx(data_in_used);
        X_eq = conj(H_d) .* Y_data_nat ./ (abs(H_d).^2 + 1/EsN0_lin + eps);
        X_eq(~isfinite(X_eq)) = 0;
        rx = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_comb_fx = err_comb_fx + sum(tx_bits_used ~= rx);

        H_d = H_est_ls(data_in_used);
        X_eq = conj(H_d) .* Y_data_nat ./ (abs(H_d).^2 + 1/EsN0_lin + eps);
        X_eq(~isfinite(X_eq)) = 0;
        rx = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_ls = err_ls + sum(tx_bits_used ~= rx);

        total_bits = total_bits + length(tx_bits_used);
    end

    BER_comb_adaptive(i_snr) = err_comb_ad / total_bits;
    BER_comb_fixed(i_snr)    = err_comb_fx / total_bits;
    BER_ls(i_snr)            = err_ls      / total_bits;

    fprintf('t=%.1f с | CombAd: %.3e | CombFx: %.3e | LS: %.3e\n', ...
            toc, BER_comb_adaptive(i_snr), BER_comb_fixed(i_snr), ...
            BER_ls(i_snr));
end

%% ГРАФИК BER
figure('Position', [60 60 900 600]);
semilogy(SNRdB, BER_pred, 'k-', 'LineWidth', 2.5, ...
    'DisplayName', 'Теория');
hold on;

valid_ls  = BER_ls > 0;
valid_fx  = BER_comb_fixed > 0;
valid_ad  = BER_comb_adaptive > 0;

semilogy(SNRdB(valid_ls), BER_ls(valid_ls), 'b-o', 'LineWidth', 1.8, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', ...
    'DisplayName', 'LS + линейная интерполяция');
semilogy(SNRdB(valid_fx), BER_comb_fixed(valid_fx), 'g--s', ...
    'LineWidth', 1.6, 'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
    'DisplayName', sprintf('Comb-DFT, фикс. L=%d', L_trunc_fixed));
semilogy(SNRdB(valid_ad), BER_comb_adaptive(valid_ad), 'r-s', ...
    'LineWidth', 1.8, 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
    'DisplayName', sprintf('Comb-DFT, адапт. (k=%d, m=%d)', ...
                           k_noise_trunc, margin_trunc));

grid on;
xlabel('SNR (дБ)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title(sprintf('BER: LS vs Comb-DFT, TDL-E (DS=%d нс)', delay_spread_ns), ...
      'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 11);
axis([min(SNRdB) max(SNRdB) 1e-6 1]);

text(min(SNRdB)+0.3, 2e-6, sprintf(['Ncomb=%d, Sf=%d\n', ...
    'TDL-E, DS=%d нс, K=%d дБ\nL_{fix}=%d'], ...
    Ncomb_ext, Sf, delay_spread_ns, K_dB, L_trunc_fixed), ...
    'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

fprintf('\nГотово!\n');