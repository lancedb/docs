---
title: Deployment
---

# Available Deployment Models

There are two deployment models you can choose from for LanceDB Enterprise: *_Managed_* and *_BYOC_*.
In each case AWS, GCP and Azure clouds are supported.

### Managed Deployment

**Managed deployment** is a private deployment of LanceDB Enterprise.
All applications will run cloud accounts managed by LanceDB in the same location as your client applications.

This hands-off approach is recommended for users who do not wish to manage the infrastructure themselves.

To access your deployment, LanceDB can provision a public or private load balancer. 

### Bring Your Own Cloud (BYOC)

With this deployment model LanceDB Enterprise is installed into your own cloud account.

This approach is recommended when:
- Users' security requirements for data residency preclude them from having data leave their account
- Other applications may be accessing the object storage

To deploy, an identity will be provisioned in your account with permissions to manage the infrastructure.

## Get Started

LanceDB Enterprise installation is highly configurable and customizable to your needs.
To get started, contact us at [contact@lancedb.com](mailto:contact@lancedb.com) for further instructions.
