



function addnoise
% Add Gaussian noise to the magnetic total-field anomaly.
% Jiajia Sun, 07/25/2017.

% fid1 = fopen('obs.dat','r');
fid1 = fopen('obs_mag.dat','r');

tmp = fscanf(fid1,'%f  %f  %f \n',[1,3]);
inc = tmp(1);
dec = tmp(2);
geomag = tmp(3);

fscanf(fid1,'%f  %f  %d \n',[1,3]);
Num = fscanf(fid1,'%d \n',1);

fid2 = fopen('obs_mag_GaussNoise0p5.dat','w');
fprintf(fid2,'%f  %f  %f \n',[inc,dec,geomag]);
fprintf(fid2,'%f  %f  %d \n',[inc,dec,1]);
fprintf(fid2,'%d \n',Num);

Data = zeros(Num,5);

for i = 1:Num
    Data(i,:) = fscanf(fid1,'%f  %f  %f  %f  %f \n', [1,5]);         
end

Data_noise = Data;
NoiseLevel = 0.5; % 2 nano-Tesla
Data_noise(:,5) = 0.5;


for i = 1:Num
   % add Gaussian noise to xx component
   Data_noise(i,4) = Data(i,4) + NoiseLevel*randn(1,1);       
   fprintf(fid2,'%f  %f  %f  %f  %12.8f \n',Data_noise(i,:));       
end


fclose(fid1);
fclose(fid2);


end