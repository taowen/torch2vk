#version 460
layout(std430) buffer;
layout(set=0,binding=0) buffer restrict writeonly O{float t_output[];};
layout(set=0,binding=1) buffer restrict readonly X{float t_x[];};
layout(set=0,binding=2) buffer restrict readonly A{float t_alpha[];};
layout(set=0,binding=3) buffer restrict readonly W{float t_weight[];};
layout(set=0,binding=4) buffer restrict readonly B{float t_bias[];};
layout(set=0,binding=5) uniform restrict readonly U{ivec4 sizes;};
layout(local_size_x=128,local_size_y=1,local_size_z=1) in;
float snake(float v,float a){float aa=a+1.0e-9;float s=sin(a*v);return v+(s*s)/aa;}
void main(){uint oc=gl_GlobalInvocationID.x,to=gl_GlobalInvocationID.y,b=gl_GlobalInvocationID.z;uint si=uint(sizes.x),so=uint(sizes.y),ic=uint(sizes.z),ocn=uint(sizes.w);if(oc>=ocn||to>=so)return;uint al=uint(t_alpha.length());float acc=t_bias[oc];for(uint ii=0u;ii<ic;++ii){float a=t_alpha[ii%al];for(uint k=0u;k<4u;++k){int numer=int(to)+1-int(k);if((numer&1)!=0)continue;int ti=numer>>1;if(ti<0||ti>=int(si))continue;acc+=snake(t_x[(b*si+uint(ti))*ic+ii],a)*t_weight[(ii*ocn+oc)*4u+k];}}t_output[(b*so+to)*ocn+oc]=acc;}
