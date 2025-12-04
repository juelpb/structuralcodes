

B = 950
H = 2450

A = B * H

d_surface = 20 # mm
n_bars = 10

A_s = n_bars * 3.14 * (d_surface / 2) ** 2
A_c = B*H

A_s_min = max(0.001*A_c, 150)


print(f"""
Areal lagt inn overflatearmering: {A_s:.1f} mm²
Minimumskrav til overflatearmering: {A_s_min:.1f} mm
CHECK: {'OK' if A_s >= A_s_min else 'IKKE OK'}
""")