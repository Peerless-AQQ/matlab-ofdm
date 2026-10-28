clear; close all; clc;

params = struct(...
    'Nr', 8, 'Ni', 4, 'K', 72, ...
    'pilot_pos', round(linspace(1,72,6)), ...
    'SNR_dB', 10, 'SIR_dB', 0, ...
    'seed', 42, 'cci_type', 'QPSK');

[Y, info] = generate_signal_model(params);

fprintf('Принятый сигнал: Y ∈ C^{%d × %d}\n', size(Y,1), size(Y,2));
fprintf('Мощность полезного: %.4f\n', mean(sum(abs(info.Y_sig).^2,1)));
fprintf('Мощность CCI:       %.4f\n', mean(sum(abs(info.Y_int).^2,1)));
fprintf('Мощность шума:      %.4f\n', mean(sum(abs(info.N).^2,1)));

%% Функция создания модели

function [Y, info] = generate_signal_model(params)

    if isfield(params,'seed') && ~isempty(params.seed)
        rng(params.seed);
    end

    Nr = params.Nr;  Ni = params.Ni;  K = params.K;
    SNR_dB = params.SNR_dB;  SIR_dB = params.SIR_dB;
    pilot_pos = params.pilot_pos;

    Ps = 1;
    Pi = Ps * 10^(-SIR_dB/10);   % суммарная мощность CCI
    Pn = Ps/Nr * 10^(-SNR_dB/10);   % дисперсия шума на антенну

    h = sqrt(Ps/Nr)   * (randn(Nr,1)    + 1i*randn(Nr,1)) / sqrt(2);
    G = sqrt(Pi/(Ni*Nr)) * (randn(Nr,Ni) + 1i*randn(Nr,Ni)) / sqrt(2);

    const = (1/sqrt(2)) * [1+1i, -1+1i, 1-1i, -1-1i];
    M_qpsk = 4;

    s = const(randi(M_qpsk, 1, K));  % [1 × K]

    cci_type = 'QPSK';
    if isfield(params,'cci_type'), cci_type = params.cci_type; end
    switch cci_type
        case 'QPSK'
            I_sym = const(randi(M_qpsk, Ni, K));
        case '16QAM'
            c16 = qammod(0:15, 16, 'UnitAveragePower', true);
            I_sym = c16(randi(16, Ni, K));
        case 'Gaussian'
            I_sym = sqrt(1/2) * (randn(Ni,K) + 1i*randn(Ni,K));
    end

    s(pilot_pos) = const(randi(M_qpsk, 1, numel(pilot_pos)));
    s_dmrs = s(pilot_pos);   % известные приёмнику символы

    N = sqrt(Pn/2) * (randn(Nr,K) + 1i*randn(Nr,K));

    % Принятый сигнал
    Y_sig = h * s;
    Y_int = G * I_sym;
    Y = Y_sig + Y_int + N;

    R_in_true = G*G' + Pn*eye(Nr);

    info.h         = h;
    info.G         = G;
    info.s         = s;
    info.s_dmrs    = s_dmrs;
    info.I_sym     = I_sym;
    info.pilot_pos = pilot_pos;
    info.R_in_true = R_in_true;
    info.Y_sig     = Y_sig;
    info.Y_int     = Y_int;
    info.N         = N;
    info.Ps        = Ps;
    info.Pi        = Pi;
    info.Pn        = Pn;
    info.SNR_dB    = SNR_dB;
    info.SIR_dB    = SIR_dB;
    info.cci_type  = cci_type;
end
