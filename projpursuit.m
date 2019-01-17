function [ T,V,ppout ]=projpursuit(X,varargin)
%PROJPURSUIT  Projection Pursuit Analysis
%   T = PROJPURSUIT(X) performs projection pursuit analysis on the 
%   matrix X, using default algorithmic parameters (see below) and
%   returns the scores in T.  The matrix X is mxn (objects x variables)
%   and T is mxp (objects x scores), where the default value of p is 2.
%
%   Projection pusuit (PP) is an exploratory data analysis technique that 
%   seeks to optimize a projection index to find "interesting" projections  
%   of objects in a lower dimensional space.  In this algorithm, kurtosis 
%   (fourth statistical moment) is used as the projection index.
%
%   T = PROJPURSUIT(X,P) returns the first P projection pursuit scores.
%   Usually P is 2 or 3 for data visualization (default = 2).
%
%   T= PROJPURSUIT(X,P,GUESS) uses GUESS initial random starting points for
%   the optimization.  Larger values of GUESS decrease the likelihood of a
%   local optimum, but increase computation time.  The default value is
%   GUESS=100.
%
%   T = PROJPURSUIT(X,...,S1,S2,...) specifies algorithmic variation of 
%   the PP analysis, where S1, S2, etc. are character strings as specified
%   with the options below.
%
%      Stepwise Unvariate ('Uni') or Multivariate ('Mul') Kurtosis
%      Ordinary ('Ord') or Recentered ('Rec') Kurtosis
%      Orthogonal Scores ('SO') or Orthogonal Loadings ('VO')
%      Minimization ('Min') or Maximization ('Max') of Kurtosis
%      Shifted ('Sh') or Standard ('St') Optimization Method
%
%   In each case, the default option is the first one.  These variations
%   are discussed in more detail below under the heading 'Algorithms'.
%
%   [T,V] = PROJPURSUIT(...) returns the P loading vectors in V (nxp).
%
%   [T,V,PPOUT] = PROJPURSUIT(...) returns additional outputs from the PP 
%   analysis in the structured variable PPOUT. These vary with the
%   algorithm selected, as indicated below.
%        PPOUT.K:  Kurtosis value(s) for the optimum subspace. Can 
%                  otherwise be found by searching for the max/min of 
%                  PPOUT.kurtObj. For multivariate methods, this is a
%                  scalar; for univariate methods, it is a 1xP vector
%                  corresponding to the optimum value in each step.
%        PPOUT.kurtObj: Kurtosis values for different initial guesses.
%        PPOUT.convFlag: Convergence status for different initial guesses.
%        PPOUT.W:  If the scores are made orthogonal for univariate
%                  methods, W and P are intermediate matrices in the 
%                  calculation of deflated matrices. The loadings are not 
%                  orthogonal in this case and are given by V=W*inv(P'*W). 
%                  If the projection vectors are set to be orthogonal, or
%                  multivariate algorithms are used, these are not 
%                  calculated.
%        PPOUT.P:  See PPOUT.W. 
%        PPOUT.Mu: The estimated row vector subtracted from the data 
%                  set, X, for re-centered methods.
%
%   Algorithms:
%
%   Univariate vs. Multivariate
%      In the stepwise univariate PP algorithm, univariate kurtosis is 
%      optimized as the projection vectors are extracted sequentially, 
%      with deflation of the original matrix at each step. In the 
%      multivariate algorithm, multivariate kurtosis is optimized as
%      all of the projection vectors are calculated simultaneously. 
%      Univariate is best for small numbers of balanced clusters that can
%      be separated in a binary fashion and runs faster than the
%      multivariate algorithm.
%
%   Minimization vs Maximization
%      Minimization of kurtosis is most often used to identify clusters.
%      Maximization may be useful in identifying outliers. Maximization
%      is not an option for recentered algorithms.
%
%   Orthogonal Scores vs. Orthogonal Loadings
%      This option is only applicable to stepwise univariate algorithms
%      for P>1 and relates to the deflation of the data matrix in the 
%      stepwise procedure. Orthogonal scores are generally preferred, 
%      since these avoid correlated scores in multiple dimensions.
%      However, the projection vectors (loadings) will not be orthogonal
%      in this case.  For multivariate methods, the loadings are always
%      orthogonal.
%
%   Ordinary vs. Recentered Algorithms
%      For data sets that are unbalanced (unequal number of members in each
%      class, the recentered algorithms may provide better results than
%      ordinary PP.
%
%   Shifted vs. Standard Algorithms
%      This refers to the mathematics of the quasi-power method. The
%      shifted algorithm should be more stable, but the option for the
%      standard algorithm has been retained. The choice is not available
%      for recentered algorithms, and the shifted algorithm may still be
%      implemented if solutions become unstable.

%%
%                             Version 1.0
%
% Original algorithms written by Siyuan Hou.
% Additional modifications made by Peter Wentzell, Chelsi Wicks, and Steve Driscoll.
%

%% Set Default Parameters
MaxMin='Min';
StSh='Sh';
VSorth='SO';
Meth='Uni';
CenMeth='Ord';
p=2;
guess=100;
ppout.W=[];
ppout.P=[];
ppout.Mu=[];

%% Check for valid inputs and parse as required

if ~exist('X','var')
    error('PP:DefineVar:X','Provide data matrix X')
elseif ~isa(X,'double')
    error('PP:InvalVar:X','Invalid data matrix X')
end

% Extract numeric variables if present
opt_start=1;       % Marks beginning of algorithmic options in varargin
if nargin>1
    if isa(varargin{1},'double')   % Second argument is p?
        p=round(varargin{1});
        opt_start=2;
        if nargin>2
            if isa(varargin{2},'double')  % No. of guesses given?
                guess=round(varargin{2});
                opt_start=3;
            end
        end
    end
end

% Check numeric variables
[m,n]=size(X);         % Check numeric variables
if numel(p)~=1 || p<1  % Check if p is valid
    error('PP:InvalVar:p','Invalid value for subspace dimension.')
elseif numel(guess)~=1 || guess<1    % Check no. of guesses
    error('PP:InvalVar:guess','Invalid value for number of guesses.')
elseif m<(p+1) || n<(p+1)   % Check X
    error('PP:InvalVar:X','Insufficient size of data matrix.')
end

% Extract string variables if present
Allowd_opts='unimulordrecsovominmaxshst';
OptStrg='';                       % String to concatenate all options
for i=opt_start:size(varargin,2)
    if ischar(varargin{i})
        temp=lower(varargin{i});
        if isempty(strfind(Allowd_opts,temp))
            error('PP:InvalVar:OptStrg','Invalid option syntax.')
        end
        OptStrg=strcat(OptStrg,temp); %creates string of all character options
    else
        error('PP:InvalVar:OptStrg','Invalid option syntax.')
    end
end

% Set options for algorithm

if strfind(OptStrg,'max')
    if strfind(OptStrg,'min')
        error('PP:InvMode:MaxMin','Choose either to minimize or maximize.')
    elseif strfind(OptStrg,'rec')
        error('PP:InvMode:MaxMin','Maximization not available for recentered PP.')
    else
        MaxMin='Max';
    end
end

if strfind(OptStrg,'st') 
    if strfind(OptStrg,'sh')
        error('PP:InvMode:StSh','Choose either the standard or shifted method')
    else
        StSh='St';
    end
end

if strfind(OptStrg,'vo')
    if strfind(OptStrg,'so')
        error('PP:InvMode:VSorth','Choose for either the scores or the projection vectors to be orthogonal')
    else
        VSorth='VO';
    end
end

if strfind(OptStrg,'mul')
    if strfind(OptStrg,'uni')
        error('PP:InvMode:UniMul','Choose either univariate or multivariate method')
    else
        Meth='Mul';
    end
end
    
if strfind(OptStrg,'rec')
    if strfind(OptStrg,'ord')
        error('PP:InvMode:OrdRec','Choose either the ordinary or recentred method')
    else
        CenMeth='Rec';
    end
end

%% Carry out PP using appropriate algorithm 

if strcmp(Meth,'Mul')
    if strcmp(CenMeth,'Rec')
        disp('Performing recentered multivariate PP')  % Diagnostic
        [T,V,R,K,Vall,kurtObj,convFlag]=rcmulkurtpp(X,p,guess);
        ppout.K=K;
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
        ppout.Mu=R;
    else
        disp(['Performing ordinary multivariate PP(' StSh ')'])  % Diagnostic
        [T,V,Vall,kurtObj,convFlag]=mulkurtpp(X,p,guess,MaxMin,StSh);
        ppout.K=min(kurtObj);
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
    end
else
    if strcmp(CenMeth,'Rec')
        disp(['Performing recentered univariate PP(' VSorth ')'])  % Diagnostic
        [T,V,R,W,P,kurtObj,convFlag]=rckurtpp(X,p,guess,VSorth);
        ppout.K=min(kurtObj);
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
        ppout.W=W;
        ppout.P=P;
        ppout.Mu=R;
    else
%         disp(['Performing ordinary univariate PP(' StSh ',' VSorth ')'])  % Diagnostic
        [T,V,W,P,kurtObj,convFlag]=okurtpp(X,p,guess,MaxMin,StSh,VSorth);
        ppout.K=min(kurtObj);
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
        ppout.W=W;
        ppout.P=P;
    end
end


%% Original Univariate Kurtosis Projection Pursuit Algorithm
function [T,V,W,P,kurtObj,convFlag]=okurtpp(X,p,guess,MaxMin,StSh,VSorth)
%% Quasi-power methods to optimize univariate kurtosis
%
%%
% Input:
%       X:       The data matrix. Rows denote samples, and columns denote variables.
%       p:       The number of projection vectors to be extracted.
%       guess:   The number of initial guesses for optimization,e.g. 100.
%                The more dimensions, the better to have more initial guesses.
%       MaxMin:  A string indicating to search for maxima or minima of kurtosis.
%                The available choices are "Max" and "Min".
%                   "Max": To search for maxima of kurtosis
%                   "Min": To search for minima of kurtosis
%                Projections revealing outliers tend to have a maximum
%                kurtosis, while projections revealing clusters tend to
%                have a minimum kurtosis. Maximization seems more important
%                in ICA to look for independent source signals, while
%                minimization appears useful in PP to looks for clusters.
%       StSh:    A string indicating if the standard or the shifted algorithm
%                is used. The available choices are "St" and "Sh".
%                   "St": To use the standard quasi-power method.
%                   "Sh": To use the shifted quasi-power method.
%       VSorth:  A string indicating whether the scores or projection
%                vectors are orthogonal. The available choices are
%                   "VO": The projection vectors are orthogonal, but
%                         scores are not, in general.
%                   "SO": The scores are orthogonal, but the projection
%                         vectors are not, in general.
%                If not specified (empty), the scores are made orthogonal.
% Output:
%       T:        Scores.
%       V:        Projection vectors.
%       W & P:    If the scores are made orthogonal, they appear in the 
%                 deflation steps. They can be used to calculate the final
%                 projection vectors with respect to the original matrix.
%                 If the projection vectors are set to be orthogonal, they
%                 are not needed. 
%       kurtObj:  Kurtosis values for different initial guesses.
%       convFlag: Convergence status for different initial guesses.

%% Note:
%
% The scores orthogonality is based on mean-centered data. If the data 
% are not mean-centered, the mean scores are added to the final scores and 
% therefore the final scores may not be not orthogonal.
% 
% For minimization of kurtosis, the standard method (st) may not be stable 
% when the number of samples is only slightly larger than the number of 
% variables. Thus, the shifted method (sh) is recommended. 

% Author: 
% S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2012.
%
% Version, Nov. 2012. This is the updated version. The original version was
% reported in the literature: S. Hou, and P. D. Wentzell, Fast and Simple 
% Methods for the Optimization of Kurtosis Used % as a Projection Pursuit 
% Index, Analytica Chimica Acta, 704 (2011) 1-15. 
%%
if exist('VSorth','var')
    if (strcmpi(VSorth,'VO')||strcmpi(VSorth,'SO'))
        % Pass
    else
        error('Please correctly choose the orthogonality of scores or projection vectors.')
    end
else
    VSorth='SO';
end
%
if strcmpi(StSh,'St') || strcmpi(StSh,'Sh')
    StSh0=StSh;
else
    error('Please correctly choose "St" or "Sh" method.')
end

%%  Mean center the data and reduce the dimensionality of the data
% if the number of variables is larger than the number of samples.
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig; 
rk=rank(X);
if p>rk
    p=rk;
    display('The component number larger than the data rank is ignored.');
end

[Uorig,Sorig,Worig]=svd(X,'econ');
X=Uorig*Sorig;
X=X(:,1:rk);
Worig=Worig(:,1:rk);
X0=X;
%% Initial settings
[r,c]=size(X);
maxcount=10000;
convFlag=cell(guess,p);
kurtObj=zeros(guess,p);
T=zeros(r,p);
W=zeros(c,p);
P=zeros(c,p);
%%
for j=1:p
    cc=c+1-j;
    convlimit=(1e-10)*cc;         % Set convergence limit
    wall=zeros(cc,guess); 
    [U,S,Vj]=svd(X,'econ'); 
    Vj=Vj(:,1:cc);                % This reduces the dimensionality of the data
    X=X*Vj;                       % when deflation is performed.
    if strcmpi(MaxMin,'Max')      % Option to search for maxima.
        invMat2=1./diag(X'*X);    % Note X'*X is diagonal due to SVD previously
    elseif strcmpi(MaxMin,'Min')  % Option to search for minima.
        Mat2=diag(X'*X); 
        VM=zeros(cc*cc,r);        % This is used to calculate "Mat1a" later
        for i=1:r
            tem=X(i,:)'*X(i,:);
            VM(:,i)=reshape(tem,cc*cc,1);
        end
    else
        error('Please correctly choose to maximize or minimize the kurtosis.')        
    end
%% Loop for different initial guesses of w  
    for k=1:guess
        w=randn(cc,1);   % Random initial guess of w for real numbers
        w=w/norm(w);
        oldw1=w;
        oldw2=oldw1;
        StSh=StSh0;
        count=0;
        while 1
            count=count+1;
            x=X*w; 
%% Maximum or minimum search      
            if strcmpi(MaxMin,'Max')         % Option to search for maxima.
                w=invMat2.*(X'*(x.*x.*x));
            elseif strcmpi(MaxMin,'Min')     % Option to search for minima.
                Mat1=sum(VM*(x.*x),2);
                Mat1=reshape(Mat1,cc,cc);
                w=Mat1\(Mat2.*w);
            end
%% Test convergence
            w=w/norm(w);
            L1=(w'*oldw1)^2;
            if (1-L1) < convlimit  
                convFlag(k,j)={'Converged'};
                break   % Exit the "while ... end" loop if converging
            elseif count>maxcount
                convFlag(k,j)={'Not converged'};
                break   % Exit if reaching the maximum iteration number
            end          
%% Continue the interation if "break" criterion is not reached    
            if strcmpi(StSh,'Sh')            % Shifted method
                w=w+0.5*oldw1;
                w=w/norm(w);
            elseif strcmpi(MaxMin,'Min')     % "St" method & minimization
                L2=(w'*oldw2)^2;             % If "St" method is not stable,
                if L2>L1 && L2>0.99          % change to shifted method
                    StSh='Sh';               
                    display('Warning: "St" method is not stable. Change to shifted method.');
                end
                oldw2=oldw1;
            end                 % "St" method & maximization: do nothing 
            oldw1=w;
        end
%% Save the projection vectors for different initial guesses
        wall(:,k)=w;
    end
%% Find the best solution from different initial guesses  
    kurtObj(:,j)=kurtosis(X*wall,1,1);
    if strcmpi(MaxMin,'Max')        % Find the best projection vector for maximum search.
        [tem,ind]=max(kurtObj(:,j));
    elseif strcmpi(MaxMin,'Min')    % Find the best projection vector for minimum search.
        [tem,ind]=min(kurtObj(:,j));
    end
    Wj=wall(:,ind);                 % Take the best projection vector as the solution.

%% Deflation of matrix 
    if strcmpi(VSorth,'VO')       % This deflation method makes the
        t=X*Wj;                   % projection vectors orthogonal.
        T(:,j)=t;
        W(:,j)=Vj*Wj;
        X=X0-X0*W*W';
    elseif strcmpi(VSorth,'SO') % This deflation method makes the scores orthogonal.
        t=X*Wj;       % This follows the deflation method used in the non-linear partial
        T(:,j)=t;     % least squares (NIPALS), which is well-known in chemometrics.
        W(:,j)=Vj*Wj;
        Pj=X'*t/(t'*t);
        P(:,j)=Vj*Pj;
        X=X0-T*P';    % This uses the Gram-Schmidt process for complex-valued vectors          
    end
end
%% Transform back into original space
W=Worig*W;         % The projection vector(s) are tranformed into original space. 
if strcmpi(VSorth,'VO')
    V=W;
    W=[];
    P=[];
    T=T+Morig*V;    % Adjust the scores. Mean scores are added.
else
    P=Worig*P;      % Vectors in P are tranformed into original space.
    V=W*inv(P'*W);  % Calculate the projection vectors by V=W*inv(P'*W)
    T=T+Morig*V;    % Adjust the scores. Mean scores are added.
    tem=sqrt(sum(abs(V).^2));
    V=V./(ones(size(V,1),1)*tem); % Make the projection vectors be unit length
    T=T./(ones(size(T,1),1)*tem); % Adjust T with respect to V
    P=P.*(ones(size(P,1),1)*tem); % Adjust P with respect to V
end
%% =================== End of the function =======================
%%


%% Original Multivariate Kurtosis Projection Pursuit Algorithm
function [T,V,Vall,kurtObj,convFlag]=mulkurtpp(X,p,guess,MaxMin,StSh)
%
% Quasi-power method to optimize multivariate kurtosis.
%%
% Input:
%       X:      The data matrix.
%       p:      The dimension of the plane or heperplane (Normally, 2 or 3).
%       guess:  The number of initial guesses for optimization. 
%               The more dimension, the better to have more initial guesses.
%       MaxMin: A string indicating to search for maxima or minima of kurtosis.
%               The available choices are "Max" and "Min".
%                   "Max": To search for maxima of kurtosis
%                   "Min": To search for minima of kurtosis
%               Projections revealing outliers tend to have a maximum
%               kurtosis, while projections revealing clusters tend to
%               have a minimum kurtosis.
%       StSh:   A string indicating if the standard or the shifted algorithm
%               is used. The available choices are "St" and "Sh".
%                   "St": To use the standard quasi-power method.
%                   "Sh": To use the shifted quasi-power method.
% Output:
%       T:        Scores.
%       V:        Projection vectors.
%       Vall:     All the projection vectors found based on different initial guesses. The
%                 best projection vectors are chosen as the solutions and put in V
%       kurtObj:  Kurtosis values for different projection vectors.
%       convFlag: Convergence status for the initial guesses..
%%
%  Mean center the data and reduce the dimensionality of the data if the number
%  of variables is larger than the number of samples.
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig;
rk=rank(X);
[Uorig,Sorig,Vorig]=svd(X,'econ');
X=Uorig*Sorig;
X=X(:,1:rk);
Vorig=Vorig(:,1:rk);
[r,c]=size(X);
%%
% Initial settings
maxcount=10000;
convlimit=1e-10;
Vall=cell(1,guess);
kurtObj=zeros(1,guess);
convFlag=cell(1,guess);
%%
for k=1:guess
    V=randn(c,p); % Random initial guess of V
    V=orth(V);
    oldV=V;
    count=0;
    while 1
        count=count+1;
        A=V'*X'*X*V;
        Ainv=inv(A);
%%      
%         kurt=0;
%         Mat=zeros(c,c);
%         for i=1:r
%             scal=(X(i,:)*V*Ainv*V'*X(i,:)');
%             kurt=kurt+scal^2;
%             Mat=Mat+scal*X(i,:)'*X(i,:);
%         end
        scal=sqrtm(Ainv)*V'*X';
        scal=sqrt(sum(scal.^2,1));
        Mat=((ones(c,1)*scal).*X');
        Mat=Mat*Mat';
% The four lines replace the above loop to increase the speed.
        
%%         
        if strcmpi(MaxMin,'Max')        % Option to search for maxima.
            M=inv(X'*X)*Mat;
            if strcmpi(StSh,'St')
                V=M*V; 
            elseif strcmpi(StSh,'sh')
                V=(M+eye(c)*trace(M)/c)*V;
            else
                error('Please correctly choose to standard or shifted method.') 
            end
        elseif strcmpi(MaxMin,'Min')    % Option to search for minima.
            M=inv(Mat)*(X'*X);
            if strcmpi(StSh,'St')
                V=M*V; 
            elseif strcmpi(StSh,'sh')
                V=(M+eye(c)*trace(M)/c)*V; 
            else
                error('Please correctly choose to standard or shifted method.') 
            end
        else
            error('Please correctly choose to maximize or minimize the kurtosis.')
        end
%%
        [V,TemS,TemV]=svd(V,'econ');        % Apply SVD to find an orthonormal basis. 
        if sum((oldV-V).^2)/(c*p)<convlimit % Test convergence.
            convFlag(1,k)={'Converged'};
            break
        elseif count>maxcount
            convFlag(1,k)={'Not converged'};
            break
        end
        oldV=V;
    end
    kurtObj(1,k)=r * sum( (sum( (sqrtm(Ainv)*V'*X').^2, 1 ) ).^2 ); % Calculate kurtosis.
    %%
    [U,S,V]=svd(X*V*V','econ');
    Vall{1,k}=Vorig*V(:,1:p);
end
%%
if strcmpi(MaxMin,'Max')        % Find the best projection vector for maximum search.
    [tem,ind]=max(kurtObj(1,:));
elseif strcmpi(MaxMin,'Min')    % Find the best projection vector for minimum search. 
    [tem,ind]=min(kurtObj(1,:));
end

V=Vall{1,ind};          % Store the projection vectors
T=X*Vorig'*V+Morig*V;   % Calculate the scores.
%% =================== End of the function =======================
%%


%% Recentered Univariate Kurtosis Projection Pursuit Algorithm
function [T,V,R,W,P,kurtObj,convFlag]=rckurtpp(X,p,guess,VSorth)
%
% Algorithms for minimization of recentered kurtosis. recentered kurtosis
% is proposed as a projection pursuit index in this work, aiming to deal with
% unbalanced clusters.
%
%%
% Input:
%       X:        The data matrix.
%       p:        The number of projection vectors to be extracted.
%       guess:    The number of initial guesses for optimization. 
%                 The more dimensions, the better to have more initial guesses.
%       VSorth:   A string indicating whether the scores or projection
%                 vectors are orthogonal. The available choices are
%                   "VO": The projection vectors are orthogonal, but
%                         scores are not, in general.
%                   "SO": The scores are orthogonal, but the projection
%                         vectors are not, in general.
%                If not specified (empty), the scores are made orthogonal.
% Output:
%       T:        Scores.
%       V:        Projection vectors.
%       R:       The estimated row vector subtracted from the data set X.
%       W & P:    If users choose scores are orthogonal, they appear in the 
%                 deflation steps. They can be used to calculate the final
%                 projection vectors with respect to the original matrix X.
%                 If the projection vectors are set to be orthogonal, they
%                 are not needed. 
%       kurtObj:  Kurtosis values for different initial guesses.
%       convFlag: Convergence status for different initial guesses.

%% Note:
% Users have the option to make the projection vectors or scores orthogonal.
% The scores orthogonality is based on mean-centered data. If the data 
% are not mean-centered, the mean scores are added to the final scores and 
% therefore the final scores may not be not orthogonal.
%% Author: 
% S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2012.
%
% This algorithm is based on the Quasi-Power methods. The Quasi-power 
% methods were reported in the literature: S. Hou, and P. D. Wentzell, 
% Fast and Simple Methods for the Optimization of Kurtosis Used as a 
% Projection Pursuit Index, Analytica Chimica Acta, 704 (2011) 1-15.
%
%%
if exist('VSorth','var')
    if (strcmpi(VSorth,'VO')||strcmpi(VSorth,'SO'))
        % Pass
    else
        error('Please correctly choose the orthogonality of scores or projection vectors.')
    end
else
    VSorth='SO';
end
%
%%  Mean center the data and reduce the dimensionality of the data
% if the number of variables is larger than the number of samples.
Morig=mean(X);
X=X-ones(size(X,1),1)*Morig; 
rk=rank(X);
if p>rk
    p=rk;
    display('The component number larger than the data rank is ignored.');
end
% 
[Uorig,Sorig,Worig]=svd(X,'econ');
X=Uorig*Sorig;
X=X(:,1:rk);
Worig=Worig(:,1:rk);
X0=X;
%% Initial settings
[r,c]=size(X);
maxcount=10000;
convFlag=cell(guess,p);
kurtObj=zeros(guess,p);
T=zeros(r,p);
W=zeros(c,p);
P=zeros(c,p);
ALPH=zeros(1,p);
%%
for j=1:p
    cc=c+1-j;
    convlimit=(1e-10)*cc;         % Set convergence limit
    wall=zeros(cc,guess); 
    alphall=zeros(1,guess);
    [U,S,Vj]=svd(X,'econ'); 
    Vj=Vj(:,1:cc);                % This reduces the dimensionality of the data
    X=X*Vj;                       % when deflation is performed.  
    for k=1:guess
        w=randn(cc,1);   % Random initial guess of w for real numbers
        w=w/norm(w);
        alph=mean(X*w);
        oldw1=w;
        oldw2=oldw1;
        count=0;
        while 1
            count=count+1;
            x=X*w;
            xalph=(x-alph);
            alph=alph + sum(xalph.^3) / (3*sum(xalph.^2)); % Update alpha (alph) value
            mu=alph*w';                 % Updata mu, given w and alpha (alph)
            tem=(x-alph).^2;
            dalph_dv=(X'*tem)/sum(tem); % Calculate dalpha/dv
            tem1=X'-dalph_dv*ones(1,r); 
            tem2=X-ones(r,1)*mu;
            Mat1=((ones(cc,1)*tem').*(tem1))*(tem2);
            Mat2=tem1*tem2;
            w=Mat1\(Mat2*w);            % updata w
%% Test convergence
            w=w/norm(w);
            L1=(w'*oldw1)^2;
            if (1-L1) < convlimit  
                convFlag(k,j)={'Converged'};
                break   % Exit the "while ... end" loop if converging
            elseif count>maxcount
                convFlag(k,j)={'Not converged'};
                break   % Exit if reaching the maximum iteration number
            end  
%% Continue the interation if "break" criterion is not reached    
            L2=(w'*oldw2)^2; 
            if L2>L1 && L2>0.95
                w=w+(rand/5+0.8)*oldw1; 
                w=w/norm(w);
            end
            oldw2=oldw1;
            oldw1=w;
        end
%% Save the projection vectors for different initial guesses
        wall(:,k)=w;
        alphall(1,k)=alph;
    end
%% Take the best projection vector as the solution 
    kurtObj(:,j)=( r*sum((X*wall-ones(r,1)*alphall).^4) ./ ( (sum((X*wall-ones(r,1)*alphall).^2)) .^2) )'; 
    [tem,ind]=min(kurtObj(:,j));
    Wj=wall(:,ind);               % Take the best projection vector as the solution.
    for i=1:cc
        if Wj(i)~=0;
            signum=sign(Wj(i));      % Change the sign of w to make it unique
            break
        end
    end
    Wj=Wj*signum;
    ALPH(1,j)=alphall(1,ind)*signum;
%% Deflation of matrix 
    if strcmpi(VSorth,'VO')       % This deflation method makes the
        t=X*Wj;                   % projection vectors orthogonal.
        T(:,j)=t;
        W(:,j)=Vj*Wj;
        X=X0-X0*W*W';
    elseif strcmpi(VSorth,'SO') % This deflation method makes the scores orthogonal.
        t=X*Wj;       % This follows the deflation method used in the non-linear partial
        T(:,j)=t;     % least squares (NIPALS), which is well-known in chemometrics.
        W(:,j)=Vj*Wj;
        Pj=X'*t/(t'*t);
        P(:,j)=Vj*Pj;
        X=X0-T*P';          
    end
end
%% Transform back into original space
W=Worig*W;         % The projection vector(s) are tranformed into original space. 
if strcmpi(VSorth,'VO')
    V=W;
    W=[];
    P=[];
    T=T+ones(r,1)*Morig*V;        % Adjust the scores. Mean scores are added.
    R=ALPH*V'+Morig;
else
    P=Worig*P;      % Vectors in P are tranformed into original space.
    V=W*inv(P'*W);  % Calculate the projection vectors by V=W*inv(P'*W)
    T=T+ones(r,1)*Morig*V;        % Adjust the scores. Mean scores are added.
    R=ALPH*(P'*W)*W'+Morig;
    tem=sqrt(sum(abs(V).^2));
    V=V./(ones(size(V,1),1)*tem); % Make the projection vectors be unit length
    T=T./(ones(size(T,1),1)*tem); % Adjust T with respect to V
    P=P.*(ones(size(P,1),1)*tem); % Adjust P with respect to V
end
%% =========================== End of the function ============================
%%


%% Recentered Multivariate Kurtosis Projection Pursuit Algorithm
function [T,V,R,K,Vall,kurtObj,convFlag]=rcmulkurtpp(X,p,guess)
%
% Algorithms for minimization of re-centered multivariate kurtosis that is
% used as a project pursuit index. This algorithm aims to deal with
% unbalanced clusters (multivariate version). The effect of dimension is
% taken into account by introducing a dimension term in the constraint. 
%%
% Input:
%       X:      The data matrix. X cannot be singular.
%       p:      The dimensionality of the plane or heperplane (Normally, 2 or 3).
%       guess:  The number of initial guesses for optimization. 
% Output:
%       T:      Scores of the chosen subspace (with the lowest multivariate
%               kurtosis value).
%       V:      Projection vectors for the chosen subspace.
%       R:      The estimated row vector subtracted from the data set X.
%       K:      Multivariate kurtosis value for the chosen subspace.
%       Vall:   All the projection vectors found based on different initial guesses. The
%               best projection vectors are chosen as the solutions and put in V.
%       kurtObj:   Kurtosis values for the projection vectors of different initial guesses.
%       convFlag: Convergence status for the different initial guesses.
%
%%
% This algorithm extends the Quasi-Power methods reported in two papers:
% (1) S. Hou, and P. D. Wentzell, Fast and Simple Methods for the Optimization 
%     of Kurtosis Used as a Projection Pursuit Index, Analytica Chimica Acta, 
%     704 (2011) 1-15. (featured article)
% (2) S. Hou, and P. D. Wentzell,Re-centered Kurtosis as a Projection Pursuit
%     Index for Multivariate Data Analysis, Journal of Chemometrics, 28
%     (2014) 370-384.   (Special issue article)
%
% Author: 
% S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2014.
%
%% Mean-center the data 
[n,m]=size(X);
Morig=mean(X);
X=X-ones(n,1)*Morig; 

%% Initial settings
maxcount=10000;
convlimit=1e-10;
Vall=cell(1,guess);
rall=cell(1,guess);
kurtObj=zeros(1,guess);
convFlag=cell(1,guess);

%% Loop
for i=1:guess
    count=0;
    V=randn(m,p);       % Random initial guess of V
    V=orthbasis(V);
    oldV1=V;
    R=mean(X)';
    while 1
        count=count+1;
        
%% Update r 
        Y=(X-ones(n,1)*R'/p)*V;     % Note p is in the denominator
        invPsi=inv(Y'*Y);
        gj=diag(Y*invPsi*Y');
        Yj=Y*invPsi*(sum(Y))';
        J=(2* Y'* ((Yj*ones(1,p)).*Y) * invPsi - eye(p)*(sum(gj)+2))/p;  % Jacobian matrix
        f=sum(Y'.*(ones(p,1)*gj'),2);
        R=R-V*(J\f);                % Newton' method  

%% Update V    
% Calculate b1 and b2       
        XX=X-ones(n,1)*R';          % Note p is not in the denominator
        Z=XX*V;
        S=Z'*Z;
        invS=inv(S);
        ai=diag(Z*invS*Z');
        Z_ai=(ai*ones(1,p)).*Z;
        Si_ai=Z'*Z_ai;
        
        b1=-J'\(invS*Si_ai*invS* (sum(Z))');
        b2=-J'\(invS*(sum(Z_ai))');
        
% Calculate the 8 matrices
        Yj_b1_Yj=(Y*b1*ones(1,p)).*Y;
        Yj_b2_Yj=(Y*b2*ones(1,p)).*Y;
        Xj_gj=sum((gj*ones(1,m)).*X);
        
        M1=X'*Z*invS*Si_ai;
        M2=-Xj_gj'*b1'*S;
        M3=2*X'*Y*(invPsi*Y'*Yj_b1_Yj*invPsi*S);     % Parentheses added to speed up
        M4=-2*X'*Yj_b1_Yj*invPsi*S;
        
        M5=(X'.*(ones(m,1)*ai'))*XX;                 % Full rank
        M6=-Xj_gj'*b2'*Z'*XX;                        % Not full rank
        M7=2*X'*Y*(invPsi*Y'*Yj_b2_Yj*invPsi*Z'*XX); % Parentheses added to speed up
        M8=-2*X'*Yj_b2_Yj*invPsi*Z'*XX;

% Calculate new V
        V=(M5+M6+M7+M8)\(M1+M2+M3+M4);
        V=orthbasis(V);
        
% Test convergence
        L=abs(V)-abs(oldV1); 
        L=trace(L'*L);
        if L<convlimit*p
            convFlag(1,i)={'Converged'};
            break
        elseif count>maxcount
            convFlag(1,i)={'Not converged'};
            break
        end
        oldV1=V;      
    end
    
% Save the subspaces for different initial guesses. Note the basis of the
% subspace has been changed in accordance with PCA (mean-centered) criterion.
    kurtObj(1,i)=n*sum(diag(Z*inv(Z'*Z)*Z').^2);
    [Utem,Stem,Vtem]=svd(X*V,'econ'); % X has been mean-centered.
    Vtem=V*Vtem;
    Vall(1,i)={Vtem};  
	rall(1,i)={(R'*Vtem*Vtem')}; % r is saved as a row vector now.
end

%% Take the best projection vector as the solution 
[tem,ind]=min(kurtObj);
V=Vall{ind};
R=rall{ind};
T=X*V;
K=kurtObj(ind);

%% Add mean value
T=T+ones(n,1)*Morig*V; % Adjust the scores (The scores of mean vector are added).
R=R+Morig;             % Adjust r (mean vector is added).

%% ============== End of the function =====================

   
function [V]=orthbasis(A)
% Calculate an orthonormal basis for matix A using Gram-Schimdt process
% Reference: David Poole, Linear Algebra - A Modern Introduction,
% Brooks/Cole, 2003. pp.376.
%
% Input:
%   A: a matrix
% Output:
%   V: an orthonormal matrix

%%
c=size(A,2);
V(:,1)=A(:,1)/norm(A(:,1));
for i=2:c
    tem=A(:,i)-V*V'*A(:,i);
    V(:,i)=tem/norm(tem);
end
