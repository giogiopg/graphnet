import os
import shutil
import json
import subprocess

from snakemake.logging import logger
from snakemake.shell import shell

def generate_iRods_profile(filename, host = "ccirods.in2p3.fr", port = 5530, zone_name = "in2p3"):
    """ Generate json configuration file for iRods """
    config = {
        "irods_host": host,
        "irods_port": port,
        "irods_zone_name": zone_name,
    }
    json_object = json.dumps(config, indent = 4)

    logger.info("irods json configuration generated:")
    logger.info(json_object)

    with open(filename, "w") as f:
        f.write(json_object)

def is_resource_needed(workflow, resource):
    """ Check if the given resources appears in a job that will be run """
    jobs = [jobs.name for jobs in workflow.persistence.dag.needrun_jobs() if resource in jobs.resources.keys()]
    logger.debug(r"Jobs requiring {resource}: {jobs}")
    return len(jobs) > 0

def onstart_wrapper(workflow, config):
    """ Wrapper for onstart action """

    logger.info("-"*20 + " On start checks " + "-"*20)
    irods_need = is_resource_needed(workflow, "irods_socket")

    try:
        if irods_need:
            # iRods credentials
            logger.info("Check iRods credentialds ...")

            if config['irods_settings']['container'] is not None and os.path.exists(".irods/.irodsA") == False and config['irods_settings']['path'] is not None :
                os.makedirs('.irods', exist_ok=True)
                logger.info("Create iRods configuration files ...")
                generate_iRods_profile(".irods/irods_environment.json")
                # Call iinit from the container to create the cookie
                
                try:
                    shell(
                        """
                        cd $HOME
                        HOME="./"
                        iinit
                        """
                        ,container_img=config['irods_settings']['container'],
                        singularity_args = workflow.singularity_args)

                except subprocess.CalledProcessError as E:
                    logger.error("Error when trying call iinit. Maybe a wrong iRods password ?")
                    os.remove(".irods/.irodsA")
                    raise E
        else:
            logger.info("No jobs requiring iRods in the DAG, skip the credential check.")

    except Exception as E:
        logger.error("Failed, exit")
        raise E
    
    logger.info("-"*20 + "    Completed    " + "-"*20)
