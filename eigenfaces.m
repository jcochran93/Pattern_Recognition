%%
clear all, close all, clc

load ../DATA/allFaces.mat

%Plot an image for each person in Yale database

allPersons = zeros(n*6,m*6);
count = 1;
for i=1:6
    for j=1:6
        allPersons(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
            = reshape(faces(:,1+sum(nfaces(1:count-1))),n,m);
        count = count + 1;
    end
end

figure(1), axes('position',[0  0  1  1]), axis off
imagesc(allPersons), colormap gray

%%


%Plot each image for a specific person in Yale database 

for person = 1:length(nfaces)
    subset = faces(:,1+sum(nfaces(1:person-1)):sum(nfaces(1:person)));
    allFaces = zeros(n*8,m*8);
    
    count = 1;
    for i=1:8
        for j=1:8
            if(count<=nfaces(person)) 
                allFaces(1+(i-1)*n:i*n,1+(j-1)*m:j*m) ...
                    = reshape(subset(:,count),n,m);
                count = count + 1;
            end
        end
    end
    
    imagesc(allFaces), colormap gray    
end
%%

% We use the first 36 people for training data
trainingFaces = faces(:,1:sum(nfaces(1:36)));
avgFace = mean(trainingFaces,2);  % size n*m by 1;

% Compute eigenfaces on mean-subtracted training data
X = trainingFaces-avgFace*ones(1,size(trainingFaces,2));
[U,S,V] = svd(X,'econ');

figure, axes('position',[0 0 1 1]), axis off


imagesc(reshape(avgFace,n,m)) ; colormap gray;% Plot avg face

figure, axes('position',[0 0 1 1]), axis off

imagesc(reshape(U(:,1),n,m)) ; colormap gray; % Plot first eigenface

%%
for i=1:50 % plot the first 50 eigenfaces
    pause(0.2); % wait for 0.2 seconds
    imagesc(reshape(U(:,i),n,m)), colormap gray
end

%%

%% Now show eigenface reconstruction of image that was omitted from test set

testFace = faces(:,1+sum(nfaces(1:36))); % first face of person 37
figure, axes('position',[0 0 1 1]), axis off
imagesc(reshape(testFace,n,m)), colormap gray;

testFaceMS = testFace - avgFace;
for r=25:25:2275 
    reconFace = avgFace + (U(:,1:r)*(U(:,1:r)'*testFaceMS));
    imagesc(reshape(reconFace,n,m)), colormap gray
    title(['r=',num2str(r,'%d')]);
    pause(0.1)
end
%%

%% Project person 2 and 7 onto PC5 and PC6

P1num = 2;  % Person number 2
P2num = 7;  % Person number 7

P1 = faces(:,1+sum(nfaces(1:P1num-1)):sum(nfaces(1:P1num)));
P2 = faces(:,1+sum(nfaces(1:P2num-1)):sum(nfaces(1:P2num)));

P1 = P1 - avgFace*ones(1,size(P1,2));
P2 = P2 - avgFace*ones(1,size(P2,2));
 

% Project onto PCA modes 5 and 6
PCAmodes = [5 6];   
PCACoordsP1 = U(:,PCAmodes)'*P1;
PCACoordsP2 = U(:,PCAmodes)'*P2;

figure
plot(PCACoordsP1(1,:),PCACoordsP1(2,:),'kd','MarkerFaceColor','k')
axis([-4000 4000 -4000 4000]), hold on, grid on
plot(PCACoordsP2(1,:),PCACoordsP2(2,:),'r^','MarkerFaceColor','r')
set(gca,'XTick',[0],'YTick',[0]);
