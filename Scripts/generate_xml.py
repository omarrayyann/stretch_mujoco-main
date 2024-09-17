def generate_gripper_xml():
    xml = ''
    
    for i in range(100):
        gripper = f'''
    
    <body name="grasping_head_{i}" pos="19 91 91">
      <geom name="center_{i}" type="cylinder" size="0.003 0.015" pos="0.0 0.0 -0.028" rgba="0 0 1 1" contype="0" conaffinity="0"/>
      <geom name="cylinder_left_{i}" type="cylinder" size="0.003 0.02" pos="0.032 0 0.006" rgba="0 0 1 1" contype="0" conaffinity="0"/>
      <geom name="cylinder_right_{i}" type="cylinder" size="0.003 0.02" pos="-0.032 0 0.006" rgba="0 0 1 1" contype="0" conaffinity="0"/>
      <geom name="connecting_bar_{i}" type="box" size="0.03 0.002 0.002" pos="0 0 -0.012" rgba="0 0 1 1" contype="0" conaffinity="0"/>
    </body>
        '''
        xml += gripper
    
    return xml
def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    xml_content = generate_gripper_xml()
    save_to_file('tset.txt', xml_content)
