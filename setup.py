from setuptools import find_packages, setup

package_name = 'AGJ_zeta_rescue'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gill2ex',
    maintainer_email='gill2ex@dukes.jmu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zeta_rescue = AGJ_zeta_rescue.zeta_rescue:main'
        ],
    },
)
