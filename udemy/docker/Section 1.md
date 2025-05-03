what is docker ?
- docker is a set of platform as a service product that uses OS-level virtualization to deliver software in packages called containers. 
- Containers are isolated from one another and bundle their own software , libraries and configuration files.
- uses linux kernel features to create container above the linux systems
- aws  , gcp  , azure support docker

| docker container                                                                                      | Virtual Machine                                                                                                                     |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| with containers , instead of virtualizing the underlying computer like a vm , just the OS virtualized | a vm is an emulation of a computer system . run what appear to be many separate computers on hardware that is actually one computer |
| lightweight                                                                                           | heavyweight                                                                                                                         |
| native performance                                                                                    | limited performance                                                                                                                 |
| all containiers share the host                                                                        | each vm has its own os                                                                                                              |
| operating system virtualization                                                                       | hardware level                                                                                                                      |
| startup in miliseconds                                                                                | minutes                                                                                                                             |
| requires less memory space                                                                            | allocates req memory                                                                                                                |
| process level isolation , hence process less secure                                                   | fully isolation and hence more secure                                                                                               |
## Docker engine
#### Docker container

