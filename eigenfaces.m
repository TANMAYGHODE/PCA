%#pixel size of the image in horizontal or vertical direction (Square image is considered)
d = 64;

% Dimension of single vector d*d
D = d * d ; 

%Number of images in a folder
N = 150;

%Folders to be selected for team 20
Image=[1,3,6,7,8,35,10,12,13,21,24,25,33,36,37];

%Initialize input vector
x = zeros(D,N);

%Load each of the 10 images in the 15 folders
for k=1:15
    for n=1:10
        img = im2double(imread(fullfile('../data_assign1_group20/64x64',strcat(int2str(Image(k))), strcat(int2str(n),'.pgm'))));
        %imshow(img);

        %Convert 64*64 image into 4096*1 vector and put it in the nth column of
        %input vector x(4096*150)
        count = 1;
        for i = 1:d
            for j = 1:d
                x(count,n+((k)*10)) = img(j,i);
                count = count+1;
            end
        end
    end
end
%Now x is a D*N matrix, i.e. 4096*150 matrix, with each of the 150 images in
%each column

%Display x Vector
%disp(x);
 

%Find Covariance of xVector
%Covariance Matrix C
%Function to calculate Covariance Matrix
    
    %Initialize mean vector
    M = zeros(D,1);

    for i = 1:D
        M(i,1) = 0;
        for j = 1:N %N=150
            M(i,1) = M(i,1) + x(i,j);
        end
        M(i,1) = M(i,1)/N;  %N=150
    end
    
    %Display Mean Vector
    %disp(M);

    aat = zeros(D,D); %a*transpose(a)
    for n=1:N
        a = zeros(D,1); % x(i) - M
        for i=1:D
            a(i,1) = x(i,n) - M(i,1);
        end
        aat = aat + (a * a');
    end
    
    %Covariance Matrix C
    Cov_Mat = aat / N;

C = Cov_Mat;

%Find Eigen vectors and Eigen value of Covariance Matrix
[Q, V] = eig(C);
%disp(Q);
    
%Extract the eigenvalues from the diagonal of D using diag(D), 
%then sort the resulting vector in descending order.
[v, index] = sort(diag(V),'descend');
%Now v is a Dx1 (4096x1) matrix with the eigen values in the
%descending order. COrersponding indices of eigen vectors 
%are available in index(D*1) matrix
    

%Reconstruct Images
%L is the number of significant eigen vectors chosen for reconstruction

L = [1,10,20,40,80,160,320,640];
%img = im2double(imread(fullfile('/home/abilasha/Desktop/prml/Datasets1/64x64/8', strcat(int2str(randi(10)),'.pgm'))));
%for 5 random vectors change here
x1 = x(:,81);

 d1 = sqrt(D);
    x_shaped = zeros(d1,d1);
    cnt = 1;
    for i = 1:d1
        for j = 1:d1
            x_shaped(j,i) = x1(cnt,1);
            cnt = cnt + 1;
        end
    end
    
    x_orig = x_shaped;
subplot(3,3, 1), imshow(x_orig);

%Recreate images for each eigen value
for l = 1:length(L)
    x_rec = zeros(D,1);
    for i = 1:L(l)
        msev = Q(:,index(i,1)); %eigen vector corresponding to eigen value
        weight = x1' * msev;
        x_rec = x_rec + (weight * msev);
    end
   
    d1 = sqrt(D);
    x_shaped = zeros(d1,d1);
    cnt = 1;
    for i = 1:d1
        for j = 1:d1
            x_shaped(j,i) = x_rec(cnt,1);
            cnt = cnt + 1;
        end
    end    
    x_rec_shaped = x_shaped;
    subplot(3,3,l+1), imshow(x_rec_shaped);
end
