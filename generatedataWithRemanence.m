function generatedataWithRemanence

model = zeros(10,20,20);

model(2:5,4:8,4:8) = 0.04;
%model(2:5,4:6,6:14) = 0.03;
%model(4:7,13:17,8:12) = 0.05;

% fid = fopen('synmodel.sus','w');
% fprintf(fid,'%5.2f \n',reshape(model,4000,1));
% fclose(fid);
% pause

indicator = zeros(10,20,20);
indicator(2:5,4:8,4:8) = 1;
%indicator(2:5,4:6,6:14) = 1;
%indicator(4:7,13:17,8:12) = 2;

% model = load('2slabmod.sus');
% 
% indicator = zeros(numel(model),1);
% indicator(model==0.04) = 1;
% indicator(model==0.05) = 2;


% InduceField_incl = 65;   % inducing field inclination is 65 degrees
% InduceField_decl = -25;  % inducing field declination is -25 degrees

Rem1_incl = -5.7;   % inclination for the first remenant magnetization is -40 degrees.
Rem1_decl = -19.8;    % declination for the first remenant magnetization is 80 degrees.

%Rem2_incl = -5.7;   % inclination for the second remenant magnetization is 10 degrees
%Rem2_decl = 19.8;   % declination for the second remenant magnetization is 40 degrees.

% Rem1_incl = -60;   % inclination for the first remenant magnetization is -40 degrees.
% Rem1_decl = 75;    % declination for the first remenant magnetization is 80 degrees.
% 
% Rem2_incl = 25;   % inclination for the second remenant magnetization is 10 degrees
% Rem2_decl = 45;   % declination for the second remenant magnetization is 40 degrees.


meshf = 'mesh';
% dataf = 'obs1.loc';  % obs1.loc goes with 2slabmod.sus
dataf = 'obs.loc';  % the only difference between obs1.loc and obs2.loc is that, in obs2.loc, I used the declination and inclination from Furnas.

%mesh (same format as mag3d, but can't use the 20*50.0 format
fmsh=fopen(meshf,'r');
my=fscanf(fmsh,'%d',1);
mx=fscanf(fmsh,'%d',1);
mz=fscanf(fmsh,'%d',1);
fprintf('my=%4.0f  mx=%4.0f  mz=%4.0f \n', [my,mx,mz]);

y0=fscanf(fmsh,'%f',1); x0=fscanf(fmsh,'%f',1); z0=fscanf(fmsh,'%f',1);

dely=fscanf(fmsh,'%f',my); delx=fscanf(fmsh,'%f',mx); delz=fscanf(fmsh,'%f',mz);

dely(1:10)
delx(1:10)
delz(1:10)

model = reshape(model,mz,my,mx);
indicator = reshape(indicator,mz,my,mx);


fclose(fmsh);

%convert to the coordinates inmesh
xmod=zeros(1,mx+1);
xmod(1)=x0;
for ii=1:mx
    xmod(ii+1)=xmod(ii)+delx(ii);
end

ymod=zeros(1,my+1);
ymod(1)=y0;
for ii=1:my
    ymod(ii+1)=ymod(ii)+dely(ii);
end

zmod=zeros(1,mz+1);
zmod(1)=-z0;
for ii=1:mz
    zmod(ii+1)=zmod(ii)+delz(ii);
end
mcel=mx*my*mz;

fdat=fopen(dataf,'rt');
InduceField_incl =fscanf(fdat,'%f',1) 
InduceField_decl =fscanf(fdat,'%f',1) 
geomag=fscanf(fdat,'%f',1);
ndat=fscanf(fdat,'%d',1);
fprintf('number of data: %6.0f \n',ndat);

ydat=zeros(1,ndat);
xdat=zeros(1,ndat);
zdat=zeros(1,ndat);
% delt=zeros(1,ndat);
% stn=zeros(1,ndat);

for idat=1:ndat
    ydat(idat)=fscanf(fdat,'%f',1);
    xdat(idat)=fscanf(fdat,'%f',1);
    zdat(idat)=fscanf(fdat,'%f',1);
%    delt(idat)=fscanf(fdat,'%f',1);
%    stn(idat)=fscanf(fdat,'%f',1);
end
fclose(fdat);

zdat=-zdat;

theta = 0.0;
icel = 0;

MagData = zeros(1,ndat);

for ii=1:mx
    xb=xmod(ii); xe=xmod(ii+1);
    for jj=1:my
        yb=ymod(jj); ye=ymod(jj+1);
        for kk=1:mz
            zb=zmod(kk); ze=zmod(kk+1);
            
            if model(kk,jj,ii)~=0
                sus = model(kk,jj,ii);
                
                icel=icel+1;
%                 
                if indicator(kk,jj,ii) == 1
                    dincl = Rem1_incl;
                    ddecl = Rem1_decl;
                    [bx,by,bz]=magsus(xb,xe,yb,ye,zb,ze,theta,sus,xdat,ydat,zdat,dincl,ddecl,geomag);
                end
                %if indicator(kk,jj,ii) == 2
                %    
                %    dincl = Rem2_incl;
                %    dincl
                %    ddecl = Rem2_decl;
                %    [bx,by,bz]=magsus(xb,xe,yb,ye,zb,ze,theta,sus,xdat,ydat,zdat,dincl,ddecl,geomag);
                %end
                di = InduceField_incl*pi/180.0;
                dd = InduceField_decl*pi/180.0;
                MagData = MagData + bx*cos(di)*cos(dd)+by*cos(di)*sin(dd)+bz*sin(di);
            end
            
        end
    end
end

% MagData2 = MagData;
% save -v7.3 MagData.mat MagData

fid = fopen('obs_remanence.dat','w');
fprintf(fid,'%7.2f   %7.2f   %10.2f \n',[InduceField_incl,InduceField_decl,geomag]);
fprintf(fid,'%7.2f   %7.2f   %d \n',[InduceField_incl,InduceField_decl,1]);
fprintf(fid,'%d \n',ndat);
for i = 1:ndat
   fprintf(fid,'%8.2f  %8.2f  %8.2f  %12.8f  %12.8f \n',[ydat(i),xdat(i),-zdat(i),MagData(i),0]);
end
fclose(fid);

end

function [bx,by, bz]=magsus(xb,xe,yb,ye,zb,ze,theta,sus,x,y,z,...
                            dincl,ddecl,geomag)
%
% function [bx,by, bz]=magsus(xb,xe,yb,ye,zb,ze,theta,x,y,z,...
%                             sus,dincl,ddecl,geomag)
%
% computes the three components of magnetic field due to a
% cuboidal prism with a constant susceptibility
%
%Inputs:
% xb, xe: beginning and ending x-coord of the prism
% yb, ye: beginning and ending y-coord of the prism
% zb, ze: beginning and ending z-coord of the prism
% theta:  rotation angle of the prism w.r.t north
% sus:    susceptibility
%
% x,y,z:  observation stored in arrays
%
% dincl,ddecl: inclination and declination of the magnetization
%              (1) same direction as the inducing field if only
%                  induced magnetization is considered
%              (2) arbitrary direction to simulate total magnetization
%                  in a different direction from inducing field
% geomag:      inducing field strength: geomag and sus are used to 
%              specify intuitively the magnetization strength
%
%
% Outout:
% bx,by,bz:   magnetic anomaly (array has the same shape as x, y, or z)
%
%------------------------------------------------------------------
% Author: Yaoguo Li
%
%

fact=geomag*sus/(4.0*pi);
%
theta=theta*pi/180.0;

% rotation matrix
tran(1)=cos(theta); tran(2)=sin(theta);

% calculate the direction cosine of the geomagnetic field:
% in the rotated system which aligns with the sides of the
% prism.

di=pi*dincl/180.0;
dd=pi*ddecl/180.0;

cx=cos(di)*cos(dd-theta)*fact;
cy=cos(di)*sin(dd-theta)*fact;
cz=sin(di)*fact;

% findind the width and center of the prism:
xc=0.5*(xe+xb); xwidth=0.5*(xe-xb);
yc=0.5*(ye+yb); ywidth=0.5*(ye-yb);

xcntr= xc*tran(1) + yc*tran(2);
ycntr=-xc*tran(2) + yc*tran(1);

% observation location in the rotated coordinate system
x0= x*tran(1) + y*tran(2);
y0=-x*tran(2) + y*tran(1);

% Note: all the horizontal components are rotated to the new
% coord system. They must be rotated back to the original system.


% begin the calculation
a1=xcntr-xwidth-x0;
a2=xcntr+xwidth-x0;
b1=ycntr-ywidth-y0;
b2=ycntr+ywidth-y0;
h1=zb-z;
h2=ze-z;
     
r111=sqrt(a1.^2 + b1.^2 + h1.^2);
r112=sqrt(a1.^2 + b1.^2 + h2.^2);

r211=sqrt(a2.^2 + b1.^2 + h1.^2);
r212=sqrt(a2.^2 + b1.^2 + h2.^2);
 
r121=sqrt(a1.^2 + b2.^2 + h1.^2);
r122=sqrt(a1.^2 + b2.^2 + h2.^2);
 
r221=sqrt(a2.^2 + b2.^2 + h1.^2);
r222=sqrt(a2.^2 + b2.^2 + h2.^2);

%----1/(DyDz) term
top1=(a2 + r222);
bot1=(a1 + r122);

top2=(a1 + r121);
bot2=(a2 + r221);

top3=(a1 + r112);
bot3=(a2 + r212);

top4=(a2 + r211);
bot4=(a1 + r111);

top=top1.*top2.*top3.*top4;
bot=bot1.*bot2.*bot3.*bot4;

tyz=log(top./bot);

%----1/(DxDz) term
top1=(b2 + r222);
bot1=(b1 + r212);

top2=(b1 + r211);
bot2=(b2 + r221);

top3=(b1 + r112);
bot3=(b2 + r122);

top4=(b2 + r121);
bot4=(b1 + r111);

top=top1.*top2.*top3.*top4;
bot=bot1.*bot2.*bot3.*bot4;

txz=log(top./bot);

%----1/(DxDy) term
top1=(h2 + r222);
bot1=(h1 + r221);

top2=(h1 + r211);
bot2=(h2 + r212);

top3=(h1 + r121);
bot3=(h2 + r122);

top4=(h2 + r112);
bot4=(h1 + r111);

top=top1.*top2.*top3.*top4;
bot=bot1.*bot2.*bot3.*bot4;

txy=log(top./bot);

tol=0.0*a1 + 1.0E-10;

%----1/(DxDx) term
del=sign(a1).*max(abs(a1),tol) + (1-abs(sign(a1))).*tol;
txx=    - atan(b1.*h2./(r112.*del)) + atan(b1.*h1./(r111.*del));
txx=txx + atan(b2.*h2./(r122.*del)) - atan(b2.*h1./(r121.*del));

del=sign(a2).*max(abs(a2),tol) + (1-abs(sign(a2))).*tol;
txx=txx + atan(b1.*h2./(r212.*del)) - atan(b1.*h1./(r211.*del));
txx=txx - atan(b2.*h2./(r222.*del)) + atan(b2.*h1./(r221.*del));

%----1/(DyDy) term
del=sign(b1).*max(abs(b1),tol) + (1-abs(sign(b1))).*tol;
tyy=    - atan(a1.*h2./(r112.*del)) + atan(a1.*h1./(r111.*del));
tyy=tyy + atan(a2.*h2./(r212.*del)) - atan(a2.*h1./(r211.*del));

del=sign(b2).*max(abs(b2),tol) + (1-abs(sign(b2))).*tol;
tyy=tyy + atan(a1.*h2./(r122.*del)) - atan(a1.*h1./(r121.*del));
tyy=tyy - atan(a2.*h2./(r222.*del)) + atan(a2.*h1./(r221.*del));

%---1/(DzDz) term
tzz= - (txx + tyy);

% mag components
bx0=txx*cx + txy*cy + txz*cz;
by0=txy*cx + tyy*cy + tyz*cz;
bz=txz*cx + tyz*cy + tzz*cz;

% rotate Bx and By back to original system
bx= tran(1)*bx0 - tran(2)*by0;
by= tran(2)*bx0 + tran(1)*by0;


% end of function magsus
end
