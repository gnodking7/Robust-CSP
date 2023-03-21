% ---------------------------------------------------- %
%   rbstCSP_SCF: SCF iteration with line search
%                for solving CSP-NRQ
% ---------------------------------------------------- %

function [x, Q, X, RESD, iter, Nls] = rbstCSP_SCF(SM_MAT, SP_MAT, x0, tol)
    % Input:
    %   SM_MAT - function handle
    %            \Sigma(x) = SM_MAT(x)
    %            [\Sigma(x) \tilde{\Sigma}(x)] = SM_MAT(x)
    %   SP_MAT - function handle
    %            \Sigma(x) = SP_MAT(x)
    %            [\Sigma(x) \tilde{\Sigma}(x)] = SP_MAT(x)
    %   x0 - initial vector
    %   tol - tolerance
    % Output:
    %   x - final vector
    %   Q - array of objective values at each iterate
    %   X - matrix with columns corresponding to iterates
    %   RESD - relative NEPv residual norm at each iterate
    %   iter - number of iterations
    %   Nls - number of line searches

    if nargin < 4
        tol = 1.0E-8; 
    end

    maxit = 200;
    Q = []; X = []; RESD = [];

    % +++++++++++++++++++++++++++
    % --- MAIN LOOP
    % +++++++++++++++++++++++++++
    x = x0;
    Nls = 0; % number of line search
    for i = 1:maxit
        % Evaluate ratio
        [SM, TSM] = SM_MAT(x);
        [SP, TSP] = SP_MAT(x);
        q = (x' * SM * x) / (x' * (SM + SP) * x);
        Q = [Q, q];
        X = [X, x];

        % Evaluate relative NEPv norm
        resd = norm(SM * x - q * (SM + SP) * x) / (norm(SM * x) + q * norm((SM + SP) * x));
        RESD = [RESD, resd];

        % Stopping condition
        if i > 1 && resd < tol
            break;
        end

        % SCF updating - compute the smallest eigenpair
        A = SM + TSM; B = SP + TSP;     % B is not necessarily positive definite!
        [V, E] = eig(A, A + B);
        tt = diag(E);
        IDX = find(tt==min(tt(tt>0)));
        IDX = IDX(end);
        new_x = V(:, IDX);
        lam = tt(IDX);
        new_q = (new_x' * SM_MAT(new_x) * new_x) / (new_x' * (SM_MAT(new_x) + SP_MAT(new_x)) * new_x);

        % Linesearch
        xx = new_x / norm(new_x);
        if q ~= lam
            xx = -sign((lam - q) * (xx' * (SM + SP) * x)) * xx;
        end
        sd = -2 * (SM * x - q * (SM + SP) * x) / (x' * (SM + SP) * x);

        if new_q > q - abs(sd' * (xx - x)) * 0.01
            d = xx - x;
            d = d - x * x' * d;
            t = (d' * sd) / norm(sd) / norm(d);
            if abs(t) < 0.01
                d = d / norm(d) + 0.01 * sd / norm(sd);
            end
            d = d /norm(d);
            t = 0.01 * abs(d' * sd);
            Nls = Nls + 1;
            alpha = norm(xx - x);
            lsinfo = false;
            while alpha > 1.0E-15
                x1t = x + alpha * d;
                if alpha * t < q - (x1t' * SM_MAT(x1t) * x1t) / (x1t' * (SM_MAT(x1t) + SP_MAT(x1t)) * x1t)
                    lsinfo = true;
                    break
                end
                alpha = alpha / 10;
            end
        else
            x1t = xx;
            lsinfo = true;
        end
        if ~lsinfo
            % line search fail, possibly due to non-smoothness or numerical error 
            [~, ind] = min(Q);
            x = X(:, ind);
            break;
        end
        x = x1t;
        x = x / norm(x);
    end

    iter = i;

end
